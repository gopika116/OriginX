"""Handcrafted forensic artifact detector (D3) for OriginX_Video.

Analyzes face crops and surrounding context for:
  (a) Laplacian variance ratio (face vs background blending blur)
  (b) High-pass noise residual statistics (unnatural lack of sensor noise)
  (c) JPEG 8x8 blocking grid discontinuities

All features are explainable and deterministic (no ML models required).
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

logger = logging.getLogger("originx_video.artifact_features")


def compute_laplacian_variance(img_gray: np.ndarray) -> float:
    """Compute Laplacian variance as an indicator of focus/sharpness."""
    if img_gray is None or img_gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(np.var(lap))


def compute_noise_residual_std(img_gray: np.ndarray) -> float:
    """Extract high-pass noise residual by subtracting median filter and compute standard deviation."""
    if img_gray is None or img_gray.size == 0:
        return 0.0
    # Median filter estimates underlying smooth image; difference is sensor noise + high frequency
    smooth = cv2.medianBlur(img_gray, 3)
    residual = cv2.subtract(img_gray, smooth)
    return float(np.std(residual))


def compute_jpeg_blocking_metric(img_gray: np.ndarray) -> float:
    """Measure discontinuity across 8x8 JPEG grid boundaries versus intra-block differences.

    Natural JPEG images have characteristic peaks at 8x8 boundaries.
    Warped/resampled deepfake faces often exhibit disrupted or smoothed grids.
    """
    if img_gray is None or img_gray.shape[0] < 32 or img_gray.shape[1] < 32:
        return 0.5

    h, w = img_gray.shape
    img_f = img_gray.astype(np.float64)

    # Horizontal boundary differences (columns 8k vs 8k-1)
    boundary_diffs_h = []
    intra_diffs_h = []
    for col in range(8, w - 8, 8):
        boundary_diffs_h.append(np.abs(img_f[:, col] - img_f[:, col - 1]))
        intra_diffs_h.append(np.abs(img_f[:, col - 4] - img_f[:, col - 5]))

    # Vertical boundary differences (rows 8k vs 8k-1)
    boundary_diffs_v = []
    intra_diffs_v = []
    for row in range(8, h - 8, 8):
        boundary_diffs_v.append(np.abs(img_f[row, :] - img_f[row - 1, :]))
        intra_diffs_v.append(np.abs(img_f[row - 4, :] - img_f[row - 5, :]))

    if not boundary_diffs_h or not boundary_diffs_v:
        return 0.5

    mean_b_h = np.mean(np.concatenate(boundary_diffs_h))
    mean_i_h = np.mean(np.concatenate(intra_diffs_h)) + 1e-6

    mean_b_v = np.mean(np.concatenate(boundary_diffs_v))
    mean_i_v = np.mean(np.concatenate(intra_diffs_v)) + 1e-6

    blocking_ratio = 0.5 * ((mean_b_h / mean_i_h) + (mean_b_v / mean_i_v))
    return float(blocking_ratio)


class ArtifactDetector:
    """Handcrafted explainable artifact detector (D3)."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        blur_ratio_threshold: float = 1.8,
        noise_residual_threshold: float = 0.55,
        jpeg_blocking_threshold: float = 0.45,
    ):
        # Default weights documented in README
        self.weights = weights or {
            "blur_ratio": 0.40,
            "noise_residual": 0.35,
            "jpeg_blocking": 0.25,
        }
        self.blur_ratio_threshold = blur_ratio_threshold
        self.noise_residual_threshold = noise_residual_threshold
        self.jpeg_blocking_threshold = jpeg_blocking_threshold

    def analyze_frame_artifacts(
        self,
        raw_frame: np.ndarray,
        face_crop: np.ndarray,
        face_box: Optional[Tuple[int, int, int, int]],
        is_face_detected: bool,
    ) -> Dict[str, float]:
        """Analyze a single frame and its face crop for blending blur, noise anomalies, and blocking."""
        if not is_face_detected or face_crop is None:
            # Masked / Neutral defaults
            return {
                "laplacian_ratio": 1.0,
                "blur_anomaly": 0.0,
                "face_noise_std": 2.5,
                "bg_noise_std": 2.5,
                "noise_anomaly": 0.0,
                "jpeg_blocking_ratio": 1.0,
                "jpeg_anomaly": 0.0,
                "composite_frame_artifact": 0.0,
            }

        crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = crop_gray.shape

        # Inner face core vs perimeter ring
        margin_h, margin_w = int(h * 0.2), int(w * 0.2)
        inner_face = crop_gray[margin_h : h - margin_h, margin_w : w - margin_w]

        # Surrounding context: outer border ring of crop
        top_ring = crop_gray[:margin_h, :]
        bottom_ring = crop_gray[h - margin_h :, :]
        left_ring = crop_gray[margin_h : h - margin_h, :margin_w]
        right_ring = crop_gray[margin_h : h - margin_h, w - margin_w :]
        bg_ring = np.concatenate(
            [top_ring.flatten(), bottom_ring.flatten(), left_ring.flatten(), right_ring.flatten()]
        )

        # 1. Laplacian variance ratio
        var_face = compute_laplacian_variance(inner_face)
        var_bg = float(np.var(cv2.Laplacian(bg_ring.reshape(-1, 1), cv2.CV_64F))) if bg_ring.size > 0 else var_face

        # Ratio of face to boundary sharpness
        ratio = (var_face + 1e-4) / (var_bg + 1e-4)
        # Deepfake boundary blending causes severe blur disparity (ratio << 1.0 or ratio >> 1.0)
        # We compute anomaly as deviation from 1.0 in log space
        log_dev = abs(np.log(max(1e-2, min(100.0, ratio))))
        blur_anomaly = float(np.clip(log_dev / 2.0, 0.0, 1.0))

        # 2. High-pass noise residual
        face_noise = compute_noise_residual_std(inner_face)
        bg_noise = float(np.std(bg_ring - np.median(bg_ring))) if bg_ring.size > 0 else face_noise

        # Lack of sensor noise in synthetic crops (< 1.2 is unnaturally smooth)
        smoothness_flag = max(0.0, (1.8 - face_noise) / 1.8)
        # Discrepancy between face noise and background noise
        noise_discrepancy = abs(face_noise - bg_noise) / (face_noise + bg_noise + 1e-4)
        noise_anomaly = float(np.clip(0.5 * smoothness_flag + 0.5 * noise_discrepancy, 0.0, 1.0))

        # 3. JPEG 8x8 blocking metric
        blocking_ratio = compute_jpeg_blocking_metric(crop_gray)
        # Deviations from normal blocking ratio ~1.08 - 1.25 indicate resampling or synthetic generation
        jpeg_anomaly = float(np.clip(abs(blocking_ratio - 1.15) / 0.5, 0.0, 1.0))

        # Composite frame artifact score
        composite = (
            self.weights["blur_ratio"] * blur_anomaly
            + self.weights["noise_residual"] * noise_anomaly
            + self.weights["jpeg_blocking"] * jpeg_anomaly
        )

        return {
            "laplacian_ratio": float(ratio),
            "blur_anomaly": blur_anomaly,
            "face_noise_std": float(face_noise),
            "bg_noise_std": float(bg_noise),
            "noise_anomaly": noise_anomaly,
            "jpeg_blocking_ratio": float(blocking_ratio),
            "jpeg_anomaly": jpeg_anomaly,
            "composite_frame_artifact": float(composite),
        }

    def detect_artifacts(
        self,
        raw_frames: List[np.ndarray],
        face_crops: List[np.ndarray],
        face_boxes: List[Optional[Tuple[int, int, int, int]]],
        face_masks: List[bool],
    ) -> Dict[str, Any]:
        """Aggregate artifact forensics across all 16 sampled frames.

        Returns artifact_score in 0..1 and explainable diagnostic metrics.
        """
        valid_indices = [i for i, m in enumerate(face_masks) if m]
        if not valid_indices:
            logger.warning("No faces detected in video; returning neutral artifact baseline.")
            return {
                "artifact_score": 0.20,
                "blur_ratio_mean": 1.0,
                "noise_discrepancy_mean": 0.1,
                "jpeg_blocking_mean": 1.1,
                "sub_scores": {"blur": 0.1, "noise": 0.1, "jpeg": 0.1},
                "details": "No faces detected for boundary artifact analysis.",
            }

        frame_metrics = []
        for i in valid_indices:
            metrics = self.analyze_frame_artifacts(
                raw_frames[i], face_crops[i], face_boxes[i], face_masks[i]
            )
            frame_metrics.append(metrics)

        # Aggregate averages
        mean_blur_ratio = float(np.mean([m["laplacian_ratio"] for m in frame_metrics]))
        mean_blur_anomaly = float(np.mean([m["blur_anomaly"] for m in frame_metrics]))
        mean_noise_face = float(np.mean([m["face_noise_std"] for m in frame_metrics]))
        mean_noise_bg = float(np.mean([m["bg_noise_std"] for m in frame_metrics]))
        mean_noise_anomaly = float(np.mean([m["noise_anomaly"] for m in frame_metrics]))
        mean_jpeg_ratio = float(np.mean([m["jpeg_blocking_ratio"] for m in frame_metrics]))
        mean_jpeg_anomaly = float(np.mean([m["jpeg_anomaly"] for m in frame_metrics]))

        composite_score = (
            self.weights["blur_ratio"] * mean_blur_anomaly
            + self.weights["noise_residual"] * mean_noise_anomaly
            + self.weights["jpeg_blocking"] * mean_jpeg_anomaly
        )
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        # Check for NaNs
        if np.isnan(composite_score):
            composite_score = 0.5

        return {
            "artifact_score": round(composite_score, 4),
            "blur_ratio_mean": round(mean_blur_ratio, 3),
            "face_noise_std": round(mean_noise_face, 3),
            "bg_noise_std": round(mean_noise_bg, 3),
            "noise_discrepancy_mean": round(mean_noise_anomaly, 3),
            "jpeg_blocking_mean": round(mean_jpeg_ratio, 3),
            "sub_scores": {
                "blur_anomaly": round(mean_blur_anomaly, 4),
                "noise_anomaly": round(mean_noise_anomaly, 4),
                "jpeg_anomaly": round(mean_jpeg_anomaly, 4),
            },
            "per_frame": frame_metrics,
        }
