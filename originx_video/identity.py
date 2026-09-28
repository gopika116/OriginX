"""Face identity consistency detector (D2) for OriginX_Video.

Uses InceptionResnetV1(pretrained='vggface2') to extract 512-d facial embeddings
per frame and evaluates pairwise cosine similarity across all sampled frames.
Detects identity drift and dual-identity clusters characteristic of face swaps.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("originx_video.identity")


class IdentityDetector:
    """Evaluates facial identity stability across 16 frames using FaceNet embeddings."""

    def __init__(
        self,
        pretrained: str = "vggface2",
        cluster_dispersion_threshold: float = 0.65,
        drift_tolerance_threshold: float = 0.70,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.cluster_dispersion_threshold = cluster_dispersion_threshold
        self.drift_tolerance_threshold = drift_tolerance_threshold
        self.model = None
        self._init_model(pretrained)

    def _init_model(self, pretrained: str):
        try:
            import os
            from facenet_pytorch import InceptionResnetV1
            
            # Check local bundled weights first
            local_weights = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "models", "20180402-114759-vggface2.pt")
            )
            if pretrained == "vggface2" and os.path.exists(local_weights):
                logger.info(f"Loading local bundled FaceNet weights from {local_weights}...")
                model = InceptionResnetV1(pretrained=None, classify=False).eval().to(self.device)
                state_dict = torch.load(local_weights, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                self.model = model
                logger.info("FaceNet InceptionResnetV1 loaded from bundled weights.")
                return

            logger.info(f"Loading InceptionResnetV1(pretrained='{pretrained}') on {self.device}...")
            self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
            logger.info("FaceNet InceptionResnetV1 successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to load FaceNet InceptionResnetV1: {e}")
            self.model = None

    def preprocess_crop(self, crop_bgr: np.ndarray, target_size: int = 160) -> torch.Tensor:
        """Preprocess face crop: BGR -> RGB, resize to 160x160, normalize to [-1, 1]."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != target_size or rgb.shape[1] != target_size:
            rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        # Standard FaceNet normalization: (x - 127.5) / 128.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0
        return tensor

    def extract_embeddings(
        self,
        face_crops: List[np.ndarray],
        face_masks: List[bool],
    ) -> Tuple[np.ndarray, List[bool]]:
        """Extract (N, 512) normalized face embeddings.

        Frames without faces receive zero vectors and mask=False.
        """
        num_frames = len(face_crops)
        embeddings = np.zeros((num_frames, 512), dtype=np.float32)

        valid_indices = [i for i, m in enumerate(face_masks) if m and face_crops[i] is not None]
        if not valid_indices or self.model is None:
            return embeddings, face_masks

        # Batch process valid crops
        tensors = [self.preprocess_crop(face_crops[i]) for i in valid_indices]
        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            emb_tensor = self.model(batch)
            # L2 normalize
            emb_norm = F.normalize(emb_tensor, p=2, dim=1).cpu().numpy()

        for idx, i in enumerate(valid_indices):
            embeddings[i] = emb_norm[idx]

        return embeddings, face_masks

    def compute_pairwise_similarities(
        self,
        embeddings: np.ndarray,
        face_masks: List[bool],
    ) -> Dict[str, Any]:
        """Compute pairwise cosine similarity matrix and statistical metrics."""
        num_frames = len(embeddings)
        sim_matrix = np.dot(embeddings, embeddings.T)
        sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

        valid_indices = [i for i, m in enumerate(face_masks) if m]
        if len(valid_indices) < 2:
            logger.warning("Fewer than 2 valid face frames; returning neutral identity consistency.")
            return {
                "identity_consistency": 0.50,
                "median_similarity": 0.70,
                "mean_similarity": 0.70,
                "min_similarity": 0.70,
                "similarity_variance": 0.0,
                "dual_identity_detected": False,
                "pairwise_matrix": sim_matrix.tolist(),
                "details": "Insufficient detected faces to establish identity consistency.",
            }

        # Extract upper triangle off-diagonal pairwise similarities for valid frames
        pair_sims = []
        for i_idx, i in enumerate(valid_indices):
            for j_idx in range(i_idx + 1, len(valid_indices)):
                j = valid_indices[j_idx]
                pair_sims.append(float(sim_matrix[i, j]))

        pair_sims = np.array(pair_sims)
        median_sim = float(np.median(pair_sims))
        mean_sim = float(np.mean(pair_sims))
        min_sim = float(np.min(pair_sims))
        var_sim = float(np.var(pair_sims))

        # Dual-identity clustering check (Face Swap detection)
        # Face swaps typically produce two distinct clusters (high similarity within cluster,
        # low similarity across clusters, causing bimodal distribution and high variance).
        low_sim_ratio = float(np.mean(pair_sims < self.cluster_dispersion_threshold))
        high_sim_ratio = float(np.mean(pair_sims > 0.80))
        dual_identity = bool(low_sim_ratio > 0.15 and high_sim_ratio > 0.15 and var_sim > 0.015)

        # Compute identity_consistency score in 0..1 (1.0 = same identity, 0.0 = inconsistent/swapped)
        # Baseline mapping: median similarity from [0.40, 0.90] -> [0.0, 1.0]
        base_score = float(np.clip((median_sim - 0.40) / 0.50, 0.0, 1.0))

        # Penalize if dual-identity or high variance detected
        if dual_identity:
            penalty = 0.40
        elif var_sim > 0.02:
            penalty = 0.20
        else:
            penalty = 0.0

        identity_consistency = round(float(np.clip(base_score - penalty, 0.0, 1.0)), 4)

        if np.isnan(identity_consistency):
            identity_consistency = 0.5

        return {
            "identity_consistency": identity_consistency,
            "median_similarity": round(median_sim, 4),
            "mean_similarity": round(mean_sim, 4),
            "min_similarity": round(min_sim, 4),
            "similarity_variance": round(var_sim, 5),
            "dual_identity_detected": dual_identity,
            "pairwise_matrix": sim_matrix.tolist(),
            "valid_faces_count": len(valid_indices),
        }

    def analyze_identity(
        self,
        face_crops: List[np.ndarray],
        face_masks: List[bool],
    ) -> Dict[str, Any]:
        """Convenience method combining embedding extraction and pairwise similarity analysis."""
        embeddings, masks = self.extract_embeddings(face_crops, face_masks)
        res = self.compute_pairwise_similarities(embeddings, masks)
        res["embeddings_shape"] = list(embeddings.shape)
        return res
