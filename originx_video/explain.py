"""Explainability and reasoning engine for OriginX_Video.

Pure rule-based mapping from the 4 detector scores and sub-signals into:
  - Overall verdict (FAKE, REAL, UNCERTAIN)
  - Calibrated confidence and fake probability
  - 2-4 sentence human-readable summary
  - 3-8 structured evidence bullets with quantitative values
  - Specific fake_type_hypothesis
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger("originx_video.explain")


class ExplainabilityEngine:
    """Reasoning engine mapping forensic evidence into verdicts and explanations."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold_fake: float = 0.60,
        threshold_real: float = 0.40,
    ):
        self.weights = weights or {
            "vit_temporal": 0.40,
            "identity_consistency": 0.25,
            "artifact_score": 0.15,
            "temporal_consistency": 0.20,
        }
        self.threshold_fake = threshold_fake
        self.threshold_real = threshold_real

    def compute_composite_score(
        self,
        vit_score: float,
        identity_score: float,
        artifact_score: float,
        temporal_score: float,
    ) -> float:
        """Compute composite fake probability in 0..1 from 4 detectors.

        Higher score = higher likelihood of manipulation.
        """
        # vit_score: 0..1 (higher = fake)
        # identity_score: 0..1 (1 = real/same identity, 0 = swap/fake)
        # artifact_score: 0..1 (higher = fake)
        # temporal_score: 0..1 (1 = real/natural motion, 0 = fake/jitter)
        f_vit = float(np.clip(vit_score, 0.0, 1.0))
        f_ident = float(np.clip(1.0 - identity_score, 0.0, 1.0))
        f_art = float(np.clip(artifact_score, 0.0, 1.0))
        f_temp = float(np.clip(1.0 - temporal_score, 0.0, 1.0))

        composite = (
            self.weights["vit_temporal"] * f_vit
            + self.weights["identity_consistency"] * f_ident
            + self.weights["artifact_score"] * f_art
            + self.weights["temporal_consistency"] * f_temp
        )
        return float(np.clip(composite, 0.0, 1.0))

    def evaluate_verdict(self, fake_prob: float) -> Tuple[str, float]:
        """Determine verdict (FAKE, REAL, UNCERTAIN) and calibrated confidence."""
        if fake_prob >= self.threshold_fake:
            verdict = "FAKE"
            # Confidence scales from 0.65 to 0.98 as prob increases from T1 to 1.0
            excess = (fake_prob - self.threshold_fake) / max(0.01, 1.0 - self.threshold_fake)
            confidence = 0.65 + 0.33 * excess
        elif fake_prob <= self.threshold_real:
            verdict = "REAL"
            # Confidence scales from 0.65 to 0.98 as prob decreases from T2 to 0.0
            margin = (self.threshold_real - fake_prob) / max(0.01, self.threshold_real)
            confidence = 0.65 + 0.33 * margin
        else:
            verdict = "UNCERTAIN"
            # In uncertain band, confidence reflects how balanced/conflicted detectors are
            distance_from_edge = min(fake_prob - self.threshold_real, self.threshold_fake - fake_prob)
            span = (self.threshold_fake - self.threshold_real) / 2.0
            confidence = 0.50 + 0.25 * (1.0 - (distance_from_edge / max(0.01, span)))

        return verdict, float(np.clip(confidence, 0.50, 0.99))

    def generate_explanation(
        self,
        vit_res: Dict[str, Any],
        ident_res: Dict[str, Any],
        art_res: Dict[str, Any],
        temp_res: Dict[str, Any],
        fake_prob: float,
        verdict: str,
    ) -> Dict[str, Any]:
        """Build evidence bullets, fake type hypothesis, and 2-4 sentence summary."""
        evidence: List[Dict[str, str]] = []

        # 1. ViT Temporal Evidence
        vit_p = vit_res.get("vit_temporal", 0.5)
        top_frames = vit_res.get("top_attended_frames", [0, 1, 2])
        frame_probs = vit_res.get("frame_fake_probs", [])
        peak_p = max(frame_probs) if frame_probs else vit_p
        vit_score = round(float(np.clip(vit_p, 0.0, 1.0)) * 100.0, 1)
        vit_auth = round(100.0 - vit_score, 1)
        vit_risk = "High" if vit_score >= 60.0 else ("Medium" if vit_score >= 35.0 else "Low")
        evidence.append({
            "check": "ViT Feature & Attention Modeling",
            "result": "ANOMALOUS" if vit_score >= 55.0 else "NORMAL",
            "score": vit_score,
            "authenticity_score": vit_auth,
            "risk_level": vit_risk,
            "metric_val": f"{peak_p:.2f} peak prob",
            "detail": f"Peak frame manipulation probability {peak_p:.2f}; sequence model flagged high temporal attention on frames {top_frames}." if vit_score >= 55.0 else f"Per-frame deep representations align with authentic facial distribution (mean fake probability {vit_p:.2f}).",
        })

        # 2. Identity Stability Evidence
        med_sim = ident_res.get("median_similarity", 0.90)
        var_sim = ident_res.get("similarity_variance", 0.0)
        min_sim = ident_res.get("min_similarity", 0.85)
        dual_ident = ident_res.get("dual_identity_detected", False)
        ident_auth = round(float(np.clip(med_sim, 0.0, 1.0)) * 100.0, 1)
        ident_score = round(100.0 - ident_auth, 1)
        ident_risk = "High" if (ident_score >= 40.0 or dual_ident) else ("Medium" if ident_score >= 25.0 else "Low")
        evidence.append({
            "check": "Facial Identity Consistency",
            "result": "IDENTITY DRIFT" if (dual_ident or var_sim > 0.015 or med_sim < 0.70) else "CONSISTENT",
            "score": ident_score,
            "authenticity_score": ident_auth,
            "risk_level": ident_risk,
            "metric_val": f"{med_sim:.3f} similarity",
            "detail": f"Median inter-frame identity similarity {med_sim:.3f} (min {min_sim:.3f}, variance {var_sim:.5f}); dual-identity dispersion detected." if (dual_ident or var_sim > 0.015 or med_sim < 0.70) else f"Median inter-frame facial identity similarity {med_sim:.3f} (variance {var_sim:.5f}); consistent individual across all frames.",
        })

        # 3. Handcrafted Forensics: Blurring / Blending
        blur_ratio = art_res.get("blur_ratio_mean", 1.0)
        blur_anomaly = art_res.get("sub_scores", {}).get("blur_anomaly", 0.0)
        blur_score = round(float(np.clip(blur_anomaly, 0.0, 1.0)) * 100.0, 1)
        blur_auth = round(100.0 - blur_score, 1)
        blur_risk = "High" if blur_score >= 50.0 else ("Medium" if blur_score >= 25.0 else "Low")
        evidence.append({
            "check": "Boundary Blending & Blur",
            "result": "SEAM ARTIFACTS" if (blur_anomaly > 0.45 or blur_ratio > 3.0 or blur_ratio < 0.3) else "CLEAN",
            "score": blur_score,
            "authenticity_score": blur_auth,
            "risk_level": blur_risk,
            "metric_val": f"{blur_ratio:.2f} ratio",
            "detail": f"Laplacian variance sharpness ratio {blur_ratio:.2f} indicates abnormal boundary smoothing or seam blending." if (blur_anomaly > 0.45 or blur_ratio > 3.0 or blur_ratio < 0.3) else f"Laplacian sharpness ratio between face core and boundary is {blur_ratio:.2f} (within natural expected range).",
        })

        # 4. Handcrafted Forensics: Sensor Noise
        face_noise = art_res.get("face_noise_std", 3.0)
        bg_noise = art_res.get("bg_noise_std", 3.0)
        noise_anom = art_res.get("sub_scores", {}).get("noise_anomaly", 0.0)
        noise_score = round(float(np.clip(noise_anom, 0.0, 1.0)) * 100.0, 1)
        noise_auth = round(100.0 - noise_score, 1)
        noise_risk = "High" if noise_score >= 50.0 else ("Medium" if noise_score >= 25.0 else "Low")
        evidence.append({
            "check": "Sensor Noise Residuals",
            "result": "UNNATURAL NOISE" if (face_noise < 1.0 or noise_anom > 0.50) else "NATURAL",
            "score": noise_score,
            "authenticity_score": noise_auth,
            "risk_level": noise_risk,
            "metric_val": f"{face_noise:.2f} std",
            "detail": f"Face high-pass noise standard deviation {face_noise:.2f} (background {bg_noise:.2f}) indicates synthetic smoothness." if (face_noise < 1.0 or noise_anom > 0.50) else f"Face sensor noise standard deviation {face_noise:.2f} is consistent with physical camera sensor noise.",
        })

        # 5. JPEG 8x8 Blocking Integrity
        jpeg_metric = art_res.get("jpeg_blocking_mean", 1.0)
        jpeg_anom = art_res.get("sub_scores", {}).get("jpeg_anomaly", 0.0)
        jpeg_score = round(float(np.clip(jpeg_anom, 0.0, 1.0)) * 100.0, 1)
        jpeg_auth = round(100.0 - jpeg_score, 1)
        jpeg_risk = "High" if jpeg_score >= 50.0 else ("Medium" if jpeg_score >= 25.0 else "Low")
        evidence.append({
            "check": "JPEG 8x8 Grid Discontinuity",
            "result": "GRID DISRUPTION" if jpeg_anom > 0.55 else "CONSISTENT",
            "score": jpeg_score,
            "authenticity_score": jpeg_auth,
            "risk_level": jpeg_risk,
            "metric_val": f"{jpeg_metric:.3f} ratio",
            "detail": f"JPEG DCT boundary blocking ratio {jpeg_metric:.3f} deviates from compression baseline." if jpeg_anom > 0.55 else f"JPEG DCT blocking ratio {jpeg_metric:.3f} conforms to standard discrete cosine transform grid boundaries.",
        })

        # 6. Physiological Blink Dynamics
        blink_rate = temp_res.get("blink_rate_per_min", 14.0)
        blink_count = temp_res.get("blink_count", 0)
        blink_anom = temp_res.get("blink_anomaly", 0.0)
        blink_score = round(float(np.clip(blink_anom, 0.0, 1.0)) * 100.0, 1)
        blink_auth = round(100.0 - blink_score, 1)
        blink_risk = "High" if blink_score >= 50.0 else ("Medium" if blink_score >= 25.0 else "Low")
        evidence.append({
            "check": "Eye Blink Physiological Dynamics",
            "result": "ABNORMAL BLINKING" if (blink_rate > 35.0 or blink_anom > 0.50) else "NORMAL",
            "score": blink_score,
            "authenticity_score": blink_auth,
            "risk_level": blink_risk,
            "metric_val": f"{blink_rate:.1f}/min",
            "detail": f"Blink rate {blink_rate:.1f}/min exceeds typical human physiological range (12-20/min)." if (blink_rate > 35.0 or blink_anom > 0.50) else f"Eyelid aspect ratio transitions recorded {blink_count} blink event(s) ({blink_rate:.1f}/min).",
        })

        # 7. Landmark Micro-Jitter
        jitter = temp_res.get("landmark_jitter", 0.01)
        jitter_anom = temp_res.get("jitter_anomaly", 0.0)
        jitter_score = round(float(np.clip(jitter_anom, 0.0, 1.0)) * 100.0, 1)
        jitter_auth = round(100.0 - jitter_score, 1)
        jitter_risk = "High" if jitter_score >= 50.0 else ("Medium" if jitter_score >= 25.0 else "Low")
        evidence.append({
            "check": "Landmark Micro-Stability",
            "result": "JITTER DETECTED" if jitter_anom > 0.40 else "STABLE",
            "score": jitter_score,
            "authenticity_score": jitter_auth,
            "risk_level": jitter_risk,
            "metric_val": f"{jitter:.4f} var",
            "detail": f"Rigid facial landmark displacement jitter {jitter:.4f} exceeds stability threshold (0.045)." if jitter_anom > 0.40 else f"Rigid anchor landmark displacement variance {jitter:.4f} shows natural temporal cohesion.",
        })


        # Determine Fake Type Hypothesis
        if verdict == "REAL":
            hypothesis = "No specific artifacts detected"
        elif verdict == "UNCERTAIN":
            hypothesis = "Inconclusive / Mixed signals"
        else:
            # FAKE classification: pinpoint specific generator signature
            if dual_ident or (1.0 - ident_res.get("identity_consistency", 1.0)) > 0.35:
                hypothesis = "Face swap"
            elif temp_res.get("coherence_anomaly", 0.0) > 0.50:
                hypothesis = "Lip-sync / puppeteering"
            elif art_res.get("sub_scores", {}).get("blur_anomaly", 0.0) > 0.45:
                hypothesis = "Blending artifacts at face boundary"
            elif face_noise < 1.2 or vit_p > 0.70:
                hypothesis = "AI-generated/full synthesis"
            else:
                hypothesis = "Inconsistent identity"

        # Construct 2-4 sentence human-readable summary
        if verdict == "FAKE":
            summary = (
                f"Video analysis classified this clip as FAKE with {fake_prob * 100:.1f}% manipulation probability. "
                f"The primary forensic indicator suggests {hypothesis.lower()}. "
                f"Key anomalies were detected in the attention sequence on frames {top_frames} alongside corresponding sub-detector flags."
            )
        elif verdict == "REAL":
            summary = (
                f"Video analysis classified this clip as REAL with {(1.0 - fake_prob) * 100:.1f}% authenticity probability. "
                f"No significant deepfake artifacts, identity shifts, or physiological inconsistencies were detected across the 16 sampled frames. "
                f"Facial geometry, sensor noise, and temporal kinematics remain consistent throughout."
            )
        else:
            disagreed = []
            if vit_p >= 0.50: agreed_fake = "ViT temporal model"
            if ident_res.get("identity_consistency", 1.0) < 0.70: disagreed.append("identity detector")
            if art_res.get("artifact_score", 0.0) > 0.40: disagreed.append("handcrafted artifact detector")
            if temp_res.get("temporal_consistency", 1.0) < 0.70: disagreed.append("temporal motion detector")
            summary = (
                f"Video analysis concluded with an UNCERTAIN verdict (composite manipulation probability {fake_prob * 100:.1f}%). "
                f"Detectors produced conflicting signals: while some indicators remained within natural bounds, others exhibited borderline anomalies. "
                f"Manual inspection or supplementary multi-modal review is recommended before reaching a definitive conclusion."
            )

        return {
            "summary": summary,
            "evidence": evidence,
            "fake_type_hypothesis": hypothesis,
        }
