"""Visual Explainability Module for OriginX_Video: Grad-CAM and Attention Analysis.

Generates:
  1. ViT Grad-CAM heatmaps (last-block gradients -> 14x14 map upscaled -> overlay)
     on the top 3 attended frames.
  2. Matplotlib attention weights bar chart highlighting suspicious frames.
"""

import os
import logging
from typing import List, Optional, Any, Tuple
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

logger = logging.getLogger("originx_video.gradcam")


def compute_vit_gradcam_heatmap(
    vit_model: Any,
    face_crop: np.ndarray,
    device: Any,
    classifier_weight: Optional[torch.Tensor] = None,
    classifier_bias: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Compute 14x14 Grad-CAM activation map from ViT last encoder block."""
    if face_crop is None or face_crop.size == 0:
        return np.zeros((14, 14), dtype=np.float32)

    # Preprocess crop to (1, 3, 224, 224)
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != 224 or rgb.shape[1] != 224:
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    rgb_f = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy((rgb_f - mean) / std).permute(2, 0, 1).unsqueeze(0).to(device)

    # If vit_model is available, compute layer activations and gradients
    if vit_model is not None and hasattr(vit_model, "encoder"):
        try:
            # Enable gradient computation for Grad-CAM
            tensor.requires_grad_(True)
            last_layer = vit_model.encoder.layer[-1]

            activations = []
            gradients = []

            def forward_hook(module, inp, out):
                activations.append(out[0])

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            h1 = last_layer.register_forward_hook(forward_hook)
            h2 = last_layer.register_full_backward_hook(backward_hook)

            outputs = vit_model(tensor)
            cls_token = outputs.last_hidden_state[:, 0, :]

            if classifier_weight is not None and classifier_bias is not None:
                logits = F.linear(cls_token, classifier_weight.to(device), classifier_bias.to(device))
                score = logits[0, 1] # Target class 1: Fake
            else:
                score = cls_token.sum()

            vit_model.zero_grad()
            score.backward(retain_graph=False)

            h1.remove()
            h2.remove()

            if activations and gradients:
                act = activations[0][0, 1:, :].detach() # (196, 768)
                grad = gradients[0][0, 1:, :].detach() # (196, 768)
                weights = torch.mean(grad, dim=0, keepdim=True) # (1, 768)
                cam = torch.sum(weights * act, dim=-1) # (196,)
                cam = F.relu(cam).cpu().numpy()
                heatmap = cam.reshape((14, 14))
                if heatmap.max() > heatmap.min():
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
                    return heatmap.astype(np.float32)
        except Exception as e:
            logger.debug(f"Gradient Grad-CAM computation skipped ({e}); using activation norm.")

    # High-resolution activation fallback
    try:
        with torch.no_grad():
            outputs = vit_model(tensor)
            # Patch tokens 1..196
            patches = outputs.last_hidden_state[0, 1:, :] # (196, 768)
            norms = torch.norm(patches, dim=-1).cpu().numpy() # (196,)
            heatmap = norms.reshape((14, 14))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            return heatmap.astype(np.float32)
    except Exception:
        pass

    # Frequency-based fallback
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    heatmap = cv2.resize(lap, (14, 14))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    return heatmap.astype(np.float32)


def generate_attention_plot(
    attention_weights: List[float],
    top_attended_frames: List[int],
    output_path: str,
) -> str:
    """Generate and save bar chart of temporal attention weights across the 16 frames."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_frames = len(attention_weights)
    indices = np.arange(num_frames)

    plt.figure(figsize=(9.5, 3.8), dpi=150)
    colors = ["#e74c3c" if i in top_attended_frames else "#3498db" for i in indices]

    bars = plt.bar(
        indices, attention_weights, color=colors, width=0.62, edgecolor="#2c3e50", linewidth=0.8
    )
    plt.title(
        "OriginX_Video Temporal Attention Distribution (Top 3 Suspicious Frames in Red)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Sampled Frame Index (0 - 15)", fontsize=10)
    plt.ylabel("Attention Weight", fontsize=10)
    plt.xticks(indices, [f"F{i:02d}" for i in indices], fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in top_attended_frames:
        if i < len(attention_weights):
            val = attention_weights[i]
            plt.text(
                i,
                val + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#c0392b",
            )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def generate_gradcam_overlays(
    face_crops: List[np.ndarray],
    top_frames: List[int],
    output_dir: str,
    vit_model: Optional[Any] = None,
    device: Optional[Any] = None,
    classifier_weight: Optional[torch.Tensor] = None,
    classifier_bias: Optional[torch.Tensor] = None,
) -> List[str]:
    """Generate Grad-CAM heatmaps overlaid onto the top 3 attended face crops.

    Upscales 14x14 map to 224x224, applies Jet colormap, and blends with original.
    """
    os.makedirs(output_dir, exist_ok=True)
    overlay_paths = []

    for rank, frame_idx in enumerate(top_frames[:3]):
        if frame_idx >= len(face_crops):
            continue
        crop = face_crops[frame_idx]
        if crop is None or crop.size == 0:
            crop = np.zeros((224, 224, 3), dtype=np.uint8)

        h, w = crop.shape[:2]

        heatmap_14x14 = compute_vit_gradcam_heatmap(
            vit_model=vit_model,
            face_crop=crop,
            device=device,
            classifier_weight=classifier_weight,
            classifier_bias=classifier_bias,
        )

        # Upscale 14x14 map to crop dimensions (w, h) using bicubic interpolation
        heatmap_upscaled = cv2.resize(heatmap_14x14, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_upscaled = np.clip(heatmap_upscaled, 0.0, 1.0)
        heatmap_uint8 = np.uint8(255 * heatmap_upscaled)

        # Apply Jet colormap
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Alpha blend (0.6 original face + 0.4 heatmap overlay)
        overlay = cv2.addWeighted(crop, 0.60, colormap, 0.40, 0)

        # Visual banner annotation
        banner_text = f"Top #{rank + 1} Frame {frame_idx:02d}"
        cv2.rectangle(overlay, (4, 4), (160, 26), (20, 20, 20), -1)
        cv2.putText(
            overlay, banner_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA
        )

        out_path = os.path.join(output_dir, f"gradcam_top{rank + 1}_frame_{frame_idx:02d}.jpg")
        cv2.imwrite(out_path, overlay)
        overlay_paths.append(out_path)

    return overlay_paths
