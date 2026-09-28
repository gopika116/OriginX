"""Pipeline orchestrator for OriginX_Video.

Exposes production contract:
    detect_video(video_path: str, output_dir: Optional[str] = None) -> dict
"""

import os
import time
import logging
from typing import Dict, Optional, Any
import cv2
import yaml

from .frames_faces import extract_frames_and_faces, FaceDetector
from .model_deepfake_vit import DeepfakeViTDetector
from .identity import IdentityDetector
from .artifact_features import ArtifactDetector
from .temporal import TemporalDetector
from .explain import ExplainabilityEngine
from .gradcam import generate_attention_plot, generate_gradcam_overlays

logger = logging.getLogger("originx_video.pipeline")

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))


class VideoDeepfakePipeline:
    """Singleton pipeline holding loaded model detectors for efficient reuse."""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config = self._load_config(config_path)
        logger.info("Initializing OriginX_Video pipeline components...")

        self.face_detector = FaceDetector(
            conf_threshold=self.config.get("sampling", {}).get("dnn_confidence_threshold", 0.5)
        )
        self.vit_detector = DeepfakeViTDetector()
        self.identity_detector = IdentityDetector()
        self.artifact_detector = ArtifactDetector()
        self.temporal_detector = TemporalDetector()

        ensemble_cfg = self.config.get("ensemble", {})
        weights = ensemble_cfg.get("weights", {
            "vit_temporal": 0.40,
            "identity_consistency": 0.25,
            "artifact_score": 0.15,
            "temporal_consistency": 0.20,
        })
        thresholds = ensemble_cfg.get("thresholds", {})
        t1 = thresholds.get("fake_threshold_t1", 0.60)
        t2 = thresholds.get("real_threshold_t2", 0.40)

        self.explainer = ExplainabilityEngine(
            weights=weights, threshold_fake=t1, threshold_real=t2
        )
        logger.info("OriginX_Video pipeline successfully initialized.")

    def _load_config(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}


# Module-level cached instance
_PIPELINE_INSTANCE: Optional[VideoDeepfakePipeline] = None


def get_pipeline() -> VideoDeepfakePipeline:
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = VideoDeepfakePipeline()
    return _PIPELINE_INSTANCE


def detect_video(video_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Production contract: detect_video(video_path) -> dict

    Orchestrates all 4 detectors:
      - D1: model_deepfake_vit (ViT representation + BiGRU-attention aggregator)
      - D2: identity (InceptionResnetV1 pairwise consistency)
      - D3: artifact_features (Laplacian blur, noise residual, JPEG grid)
      - D4: temporal (FaceMesh blinks, landmark jitter, mouth-head motion)

    Returns strictly conforming dictionary adhering to contract specifications.
    """
    t0 = time.time()
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Target video not found: {video_path}")

    pipeline = get_pipeline()

    # Determine explainability output directory
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.abspath(
            os.path.join(os.path.dirname(video_path), f"originx_output_{base_name}")
        )
    os.makedirs(output_dir, exist_ok=True)
    sampled_dir = os.path.join(output_dir, "sampled_frames")
    os.makedirs(sampled_dir, exist_ok=True)

    # 1. Decode & Sample 16 Frames / Detect Faces
    extracted = extract_frames_and_faces(
        video_path=video_path,
        num_frames=16,
        margin=0.25,
        min_size=48,
        target_size=224,
        detector=pipeline.face_detector,
    )

    raw_frames = extracted["raw_frames"]
    face_crops = extracted["face_crops"]
    face_masks = extracted["face_masks"]
    face_boxes = extracted["face_boxes"]

    # Save sampled frames and face crops to disk
    for idx, (frame, crop) in enumerate(zip(raw_frames, face_crops)):
        cv2.imwrite(os.path.join(sampled_dir, f"frame_{idx:02d}.jpg"), frame)
        cv2.imwrite(os.path.join(sampled_dir, f"face_crop_{idx:02d}.jpg"), crop)

    # 2. Run D1: ViT + BiGRU Temporal Detector
    vit_res = pipeline.vit_detector.detect_deepfake(face_crops, face_masks)

    # 3. Run D2: Identity Consistency Detector
    ident_res = pipeline.identity_detector.analyze_identity(face_crops, face_masks)

    # 4. Run D3: Handcrafted Artifacts Detector
    art_res = pipeline.artifact_detector.detect_artifacts(
        raw_frames, face_crops, face_boxes, face_masks
    )

    # 5. Run D4: Temporal & Motion Consistency Detector
    temp_res = pipeline.temporal_detector.analyze_temporal(
        raw_frames, face_crops, face_masks, fps=extracted["fps"]
    )

    # 6. Ensemble Reasoning & Explanation
    fake_prob = pipeline.explainer.compute_composite_score(
        vit_score=vit_res["vit_temporal"],
        identity_score=ident_res["identity_consistency"],
        artifact_score=art_res["artifact_score"],
        temporal_score=temp_res["temporal_consistency"],
    )

    verdict, confidence = pipeline.explainer.evaluate_verdict(fake_prob)

    explanation = pipeline.explainer.generate_explanation(
        vit_res=vit_res,
        ident_res=ident_res,
        art_res=art_res,
        temp_res=temp_res,
        fake_prob=fake_prob,
        verdict=verdict,
    )

    # 7. Generate Visual Explainability Assets
    top_frames = vit_res["top_attended_frames"]
    attn_plot_path = os.path.join(output_dir, "attention_weights.png")
    generate_attention_plot(vit_res["attention_weights"], top_frames, attn_plot_path)

    gradcam_paths = generate_gradcam_overlays(
        face_crops=face_crops,
        top_frames=top_frames,
        output_dir=os.path.join(output_dir, "gradcam"),
        vit_model=pipeline.vit_detector.vit_model,
        device=pipeline.vit_detector.device,
    )

    runtime_sec = round(time.time() - t0, 3)

    # 8. Assemble Exact Contract Schema
    contract_response: Dict[str, Any] = {
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "fake_probability": round(float(fake_prob), 4),
        "explanation": {
            "summary": explanation["summary"],
            "evidence": explanation["evidence"],
            "fake_type_hypothesis": explanation["fake_type_hypothesis"],
        },
        "explainability": {
            "top_attended_frames": [int(f) for f in top_frames],
            "attention_plot": attn_plot_path,
            "gradcam_frames": gradcam_paths,
            "sampled_frames_dir": sampled_dir,
        },
        "detector_scores": {
            "vit_temporal": round(float(vit_res["vit_temporal"]), 4),
            "identity_consistency": round(float(ident_res["identity_consistency"]), 4),
            "artifact_score": round(float(art_res["artifact_score"]), 4),
            "temporal_consistency": round(float(temp_res["temporal_consistency"]), 4),
        },
        "meta": {
            "frames_sampled": int(len(raw_frames)),
            "faces_detected": int(extracted["total_faces_detected"]),
            "runtime_sec": runtime_sec,
        },
    }

    return contract_response
