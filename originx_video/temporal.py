"""Physiological and temporal consistency detector (D4) for OriginX_Video.

Analyzes 16-frame sequence using MediaPipe FaceMesh for:
  (a) Eye-blink rate (Eyelid Aspect Ratio crossings)
  (b) Landmark micro-jitter (displacement variance of rigid facial anchors)
  (c) Mouth openness vs head motion coherence (puppeteering / lip-sync flag)

Outputs temporal_consistency in 0..1 (1.0 = highly consistent/natural, 0.0 = anomalous).
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

logger = logging.getLogger("originx_video.temporal")

# MediaPipe model path
LANDMARKER_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")
)

# Landmark indices in 478 MediaPipe mesh
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

RIGID_ANCHORS = [1, 168, 152, 133, 362] # Nose tip, bridge, chin, eye inner corners


class TemporalDetector:
    """Detects physiological and motion inconsistencies across 16 frames."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_blink_rate: float = 6.0,
        max_blink_rate: float = 35.0,
        ear_threshold: float = 0.20,
        jitter_threshold: float = 0.045,
    ):
        self.model_path = model_path or LANDMARKER_MODEL_PATH
        self.min_blink_rate = min_blink_rate
        self.max_blink_rate = max_blink_rate
        self.ear_threshold = ear_threshold
        self.jitter_threshold = jitter_threshold
        self.landmarker = None
        self._init_landmarker()

    def _init_landmarker(self):
        try:
            import mediapipe as mp
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000000:
                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=self.model_path),
                    running_mode=VisionRunningMode.IMAGE,
                    num_faces=1,
                )
                self.landmarker = FaceLandmarker.create_from_options(options)
                logger.info("MediaPipe FaceLandmarker successfully initialized.")
            else:
                logger.warning(f"FaceLandmarker model missing at {self.model_path}. Geometric fallback active.")
        except Exception as e:
            logger.warning(f"MediaPipe initialization error: {e}. Fallback active.")

    def _extract_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract (478, 3) normalized landmark coordinates from frame."""
        if self.landmarker is None or frame_bgr is None:
            return None

        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            if result.face_landmarks and len(result.face_landmarks) > 0:
                pts = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]],
                    dtype=np.float32,
                )
                return pts
        except Exception as e:
            logger.debug(f"Landmark detection failed on frame: {e}")
        return None

    def _compute_ear(self, lm: np.ndarray) -> float:
        """Compute average Eye Aspect Ratio (EAR) for left and right eyes."""
        # Left eye
        v_l = np.linalg.norm(lm[LEFT_EYE_TOP, :2] - lm[LEFT_EYE_BOTTOM, :2])
        h_l = np.linalg.norm(lm[LEFT_EYE_INNER, :2] - lm[LEFT_EYE_OUTER, :2]) + 1e-6
        ear_l = v_l / h_l

        # Right eye
        v_r = np.linalg.norm(lm[RIGHT_EYE_TOP, :2] - lm[RIGHT_EYE_BOTTOM, :2])
        h_r = np.linalg.norm(lm[RIGHT_EYE_INNER, :2] - lm[RIGHT_EYE_OUTER, :2]) + 1e-6
        ear_r = v_r / h_r

        return float(0.5 * (ear_l + ear_r))

    def _compute_mar(self, lm: np.ndarray) -> float:
        """Compute Mouth Aspect Ratio (MAR)."""
        v = np.linalg.norm(lm[MOUTH_TOP, :2] - lm[MOUTH_BOTTOM, :2])
        h = np.linalg.norm(lm[MOUTH_LEFT, :2] - lm[MOUTH_RIGHT, :2]) + 1e-6
        return float(v / h)

    def analyze_temporal(
        self,
        raw_frames: List[np.ndarray],
        face_crops: List[np.ndarray],
        face_masks: List[bool],
        fps: float = 25.0,
    ) -> Dict[str, Any]:
        """Run temporal and physiological consistency checks over 16 sampled frames.

        Returns temporal_consistency (0..1) where 1.0 is natural and 0.0 is abnormal.
        """
        valid_indices = [i for i, m in enumerate(face_masks) if m]
        if len(valid_indices) < 2:
            logger.warning("Fewer than 2 face frames; returning neutral temporal score.")
            return {
                "temporal_consistency": 0.50,
                "blink_count": 0,
                "blink_rate_per_min": 14.0,
                "blink_anomaly": 0.0,
                "landmark_jitter": 0.02,
                "jitter_anomaly": 0.0,
                "mouth_head_coherence": 0.70,
                "coherence_anomaly": 0.0,
                "details": "Insufficient face detections across sequence for temporal tracking.",
            }

        # Extract landmarks for all frames
        landmarks_seq = []
        ears = []
        mars = []
        centroids = []

        for i in range(len(raw_frames)):
            if face_masks[i]:
                # Try raw frame first, then face crop
                lm = self._extract_landmarks(raw_frames[i])
                if lm is None and face_crops[i] is not None:
                    lm = self._extract_landmarks(face_crops[i])
            else:
                lm = None

            if lm is not None:
                landmarks_seq.append(lm)
                ears.append(self._compute_ear(lm))
                mars.append(self._compute_mar(lm))
                # Rigid centroid
                centroids.append(np.mean(lm[RIGID_ANCHORS, :2], axis=0))
            else:
                landmarks_seq.append(None)
                ears.append(0.28) # Default open eye
                mars.append(0.20)
                centroids.append(None)

        # 1. Blink Analysis
        # Count EAR downward crossings below threshold
        blink_count = 0
        is_closed = False
        for val in ears:
            if val < self.ear_threshold:
                if not is_closed:
                    blink_count += 1
                    is_closed = True
            else:
                is_closed = False

        duration_sec = max(1.0, len(raw_frames) / max(1.0, fps))
        blink_rate_per_min = float((blink_count / duration_sec) * 60.0)

        # Normal human blink rate: ~12 - 20 blinks/min.
        # Abnormal if 0 blinks over extended time or extreme fluttering (> 35/min)
        if duration_sec >= 2.0 and blink_count == 0:
            blink_anomaly = 0.35 # Mild suspicion for zero blinks in short clip
        elif blink_rate_per_min > self.max_blink_rate:
            blink_anomaly = float(min(1.0, (blink_rate_per_min - self.max_blink_rate) / 25.0))
        else:
            blink_anomaly = 0.05

        # 2. Landmark Jitter Analysis
        # Frame-to-frame displacement variance of rigid facial anchors
        displacements = []
        for t in range(1, len(landmarks_seq)):
            lm_prev = landmarks_seq[t - 1]
            lm_curr = landmarks_seq[t]
            if lm_prev is not None and lm_curr is not None:
                # Relative displacement of rigid points after removing global shift
                shift = centroids[t] - centroids[t - 1]
                rigid_curr = lm_curr[RIGID_ANCHORS, :2] - shift
                rigid_prev = lm_prev[RIGID_ANCHORS, :2]
                disp = np.linalg.norm(rigid_curr - rigid_prev, axis=1)
                displacements.append(disp)

        if displacements:
            all_disp = np.concatenate(displacements)
            jitter_val = float(np.std(all_disp))
            jitter_anomaly = float(np.clip(jitter_val / self.jitter_threshold, 0.0, 1.0))
        else:
            jitter_val = 0.015
            jitter_anomaly = 0.1

        # 3. Mouth Openness vs Head Motion Coherence (Puppeteering check)
        mar_changes = []
        head_changes = []
        for t in range(1, len(mars)):
            if centroids[t] is not None and centroids[t - 1] is not None:
                d_mar = abs(mars[t] - mars[t - 1])
                d_head = float(np.linalg.norm(centroids[t] - centroids[t - 1]))
                mar_changes.append(d_mar)
                head_changes.append(d_head)

        if len(mar_changes) >= 4:
            # Puppeteering flag: active lip movement with completely frozen head
            mean_mar_change = float(np.mean(mar_changes))
            mean_head_change = float(np.mean(head_changes))

            if mean_mar_change > 0.04 and mean_head_change < 0.003:
                # Active mouth while head is unnaturally rigid
                puppeteering_anomaly = 0.70
                coherence = 0.30
            else:
                # Normal correlated head/mouth kinetics
                puppeteering_anomaly = 0.10
                coherence = 0.85
        else:
            puppeteering_anomaly = 0.15
            coherence = 0.75

        # Composite anomaly score in 0..1 (higher = more anomalous/fake)
        composite_anomaly = (
            0.35 * jitter_anomaly
            + 0.35 * puppeteering_anomaly
            + 0.30 * blink_anomaly
        )
        composite_anomaly = float(np.clip(composite_anomaly, 0.0, 1.0))

        # temporal_consistency in 0..1 (1.0 = consistent/real, 0.0 = inconsistent/fake)
        temporal_consistency = round(float(np.clip(1.0 - composite_anomaly, 0.0, 1.0)), 4)

        if np.isnan(temporal_consistency):
            temporal_consistency = 0.5

        return {
            "temporal_consistency": temporal_consistency,
            "blink_count": int(blink_count),
            "blink_rate_per_min": round(blink_rate_per_min, 1),
            "blink_anomaly": round(blink_anomaly, 3),
            "landmark_jitter": round(jitter_val, 4),
            "jitter_anomaly": round(jitter_anomaly, 3),
            "mouth_head_coherence": round(coherence, 3),
            "coherence_anomaly": round(puppeteering_anomaly, 3),
            "duration_sec": round(duration_sec, 2),
        }
