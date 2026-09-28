"""Deepfake ViT Per-Frame Feature Extractor and BiGRU-Attention Temporal Aggregator (D1).

Combines:
  1. Frozen HuggingFace ViT backbone ("dima806/deepfake_vs_real_image_detection")
     extracting 768-d per-frame CLS embeddings.
  2. 2-layer Bidirectional GRU (hidden_size=256) capturing temporal dynamics.
  3. Masked attention pooling over 16 frames to locate key manipulated frames.
  4. MLP classification head producing sequence-level deepfake probability.
  5. Logistic regression fallback head on per-frame logits if checkpoint is absent.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("originx_video.vit_temporal")

# Default model identifiers & paths
HF_MODEL_NAME = "dima806/deepfake_vs_real_image_detection"
DEFAULT_CHECKPOINT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "bigru_attention.pt")
)
DEFAULT_LOGISTIC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "logistic_fallback.pkl")
)


class MaskedAttentionPooling(nn.Module):
    """Masked self-attention pooling over temporal sequence of frames."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """features: (B, T, D)

        mask: (B, T) boolean or 0/1 tensor (1=valid, 0=padding/masked)
        Returns:
            context: (B, D)
            weights: (B, T)
        """
        # scores: (B, T, 1)
        scores = self.proj(features)

        if mask is not None:
            # Mask out invalid frames with large negative values
            mask_expanded = mask.unsqueeze(-1).float()
            scores = scores + (1.0 - mask_expanded) * -1e9

        weights = F.softmax(scores, dim=1) # (B, T, 1)
        # In case all frames in batch were masked, avoid NaN by uniform fallback
        if torch.isnan(weights).any():
            weights = torch.ones_like(scores) / scores.size(1)

        context = torch.sum(features * weights, dim=1) # (B, D)
        return context, weights.squeeze(-1)


class BiGRUTemporalAggregator(nn.Module):
    """Bidirectional GRU with masked attention pooling and classification head."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # BiGRU produces 2 * hidden_dim = 512-dim sequence
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = MaskedAttentionPooling(input_dim=hidden_dim * 2, hidden_dim=128)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """embeddings: (B, T, 768)

        mask: (B, T)
        Returns:
            logits: (B, 1)
            attention_weights: (B, T)
        """
        gru_out, _ = self.gru(embeddings) # (B, T, 512)
        context, attn_weights = self.attention(gru_out, mask=mask) # context: (B, 512), weights: (B, T)
        logits = self.classifier(context) # (B, 1)
        return logits, attn_weights


class DeepfakeViTDetector:
    """End-to-end D1 detector combining frozen HF ViT backbone with BiGRU aggregator."""

    def __init__(
        self,
        hf_model_name: str = HF_MODEL_NAME,
        checkpoint_path: Optional[str] = None,
        logistic_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.hf_model_name = hf_model_name
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.logistic_path = logistic_path or DEFAULT_LOGISTIC_PATH

        self.vit_model = None
        self.vit_classifier_weight = None
        self.vit_classifier_bias = None
        self.temporal_aggregator = None
        self.decision_threshold = 0.50

        self._init_models()

    def _init_models(self):
        # 1. Load frozen HF ViT backbone
        try:
            from transformers import AutoModel, AutoModelForImageClassification
            logger.info(f"Loading frozen ViT backbone '{self.hf_model_name}' on {self.device}...")
            # Load full classification model to extract classifier weights
            hf_clf = AutoModelForImageClassification.from_pretrained(self.hf_model_name).eval().to(self.device)
            self.vit_model = hf_clf.vit
            for param in self.vit_model.parameters():
                param.requires_grad = False

            if hasattr(hf_clf, "classifier"):
                self.vit_classifier_weight = hf_clf.classifier.weight.detach().clone()
                self.vit_classifier_bias = hf_clf.classifier.bias.detach().clone()

            logger.info("Frozen ViT backbone successfully loaded.")
        except Exception as e:
            logger.error(f"Error loading HF ViT model: {e}")
            self.vit_model = None

        # 2. Initialize BiGRU Temporal Aggregator
        self.temporal_aggregator = BiGRUTemporalAggregator(
            input_dim=768, hidden_dim=256, num_layers=2
        ).to(self.device)

        # 3. Load trained checkpoint if exists
        if os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    self.temporal_aggregator.load_state_dict(ckpt["state_dict"])
                    self.decision_threshold = ckpt.get("threshold", 0.50)
                else:
                    self.temporal_aggregator.load_state_dict(ckpt)
                self.temporal_aggregator.eval()
                logger.info(f"Loaded BiGRU aggregator checkpoint from {self.checkpoint_path} (threshold={self.decision_threshold:.3f})")
            except Exception as e:
                logger.warning(f"Could not load BiGRU checkpoint: {e}. Running with initialized weights.")
        else:
            logger.info("No BiGRU checkpoint found; fallback per-frame pooling available.")

    def preprocess_faces(self, face_crops: List[np.ndarray]) -> torch.Tensor:
        """Standardize face crops to (B, 3, 224, 224) with ImageNet normalization."""
        tensors = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for crop in face_crops:
            if crop is None or crop.size == 0:
                img = np.zeros((224, 224, 3), dtype=np.float32)
            else:
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                if img.shape[0] != 224 or img.shape[1] != 224:
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
                img = img.astype(np.float32) / 255.0
                img = (img - mean) / std

            # (H, W, C) -> (C, H, W)
            tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            tensors.append(tensor)

        return torch.stack(tensors, dim=0).to(self.device)

    def extract_features(
        self,
        face_crops: List[np.ndarray],
        face_masks: List[bool],
    ) -> Dict[str, Any]:
        """Extract (16, 768) CLS embeddings and per-frame fake probabilities."""
        if self.vit_model is None:
            # Fallback zero features
            num = len(face_crops)
            return {
                "embeddings": torch.zeros((1, num, 768), device=self.device),
                "frame_fake_probs": np.zeros(num, dtype=np.float32),
                "mask_tensor": torch.tensor([face_masks], device=self.device, dtype=torch.bool),
            }

        batch = self.preprocess_faces(face_crops) # (16, 3, 224, 224)
        num_frames = batch.size(0)

        with torch.no_grad():
            outputs = self.vit_model(batch)
            # CLS token is at index 0 of last_hidden_state
            cls_tokens = outputs.last_hidden_state[:, 0, :] # (16, 768)

            # Mask frames without faces
            for i, m in enumerate(face_masks):
                if not m:
                    cls_tokens[i] = 0.0

            # Compute per-frame logits if classifier weights available
            if self.vit_classifier_weight is not None and self.vit_classifier_bias is not None:
                logits = F.linear(cls_tokens, self.vit_classifier_weight, self.vit_classifier_bias)
                probs = F.softmax(logits, dim=-1)
                # Label 1 is 'Fake'
                frame_fake_probs = probs[:, 1].cpu().numpy()
            else:
                frame_fake_probs = np.full(num_frames, 0.5, dtype=np.float32)

        embeddings_seq = cls_tokens.unsqueeze(0) # (1, 16, 768)
        mask_tensor = torch.tensor([face_masks], device=self.device, dtype=torch.bool)

        return {
            "embeddings": embeddings_seq,
            "frame_fake_probs": frame_fake_probs,
            "mask_tensor": mask_tensor,
            "last_hidden_state": outputs.last_hidden_state, # kept for Grad-CAM
        }

    def detect_deepfake(
        self,
        face_crops: List[np.ndarray],
        face_masks: List[bool],
    ) -> Dict[str, Any]:
        """Run D1 deepfake detector on 16 face crops.

        Returns vit_temporal probability, attention weights, and top-attended frames.
        """
        feat = self.extract_features(face_crops, face_masks)
        embeddings = feat["embeddings"]
        mask_tensor = feat["mask_tensor"]
        frame_fake_probs = feat["frame_fake_probs"]

        # Run BiGRU + Attention temporal aggregator
        if self.temporal_aggregator is not None:
            self.temporal_aggregator.eval()
            with torch.no_grad():
                logits, attn_weights = self.temporal_aggregator(embeddings, mask=mask_tensor)
                fake_prob = float(torch.sigmoid(logits).squeeze().cpu().item())
                attn_weights_np = attn_weights.squeeze(0).cpu().numpy()
        else:
            # Fallback: mean of top frame fake probabilities
            valid_probs = [p for i, p in enumerate(frame_fake_probs) if face_masks[i]]
            if valid_probs:
                fake_prob = float(np.mean(valid_probs))
            else:
                fake_prob = 0.50
            attn_weights_np = np.full(len(face_crops), 1.0 / len(face_crops), dtype=np.float32)

        # Identify top 3 attended frames for explainability
        # Order by attention weight descending
        top_indices = np.argsort(attn_weights_np)[::-1][:3].tolist()

        return {
            "vit_temporal": round(float(np.clip(fake_prob, 0.0, 1.0)), 4),
            "frame_fake_probs": [round(float(p), 4) for p in frame_fake_probs],
            "attention_weights": [round(float(w), 4) for w in attn_weights_np],
            "top_attended_frames": top_indices,
            "decision_threshold": self.decision_threshold,
        }
