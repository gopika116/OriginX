import os
import math
import cv2
import numpy as np
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


class DeepfakeDetector:

    def __init__(self):
        self.model_name = "dima806/deepfake_vs_real_image_detection"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading deepfake detection model...")

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def _analyze_image(self, image):
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        labels = self.model.config.id2label
        real_probability = 0.0
        fake_probability = 0.0

        for index, probability in enumerate(probabilities):
            label = labels[index].lower()
            value = probability.item() * 100
            if "fake" in label:
                fake_probability = value
            elif "real" in label:
                real_probability = value

        return real_probability, fake_probability

    def compute_uncertainty(self, real_prob, fake_prob):
        p1 = max(1e-6, min(1.0 - 1e-6, real_prob / 100.0))
        p2 = max(1e-6, min(1.0 - 1e-6, fake_prob / 100.0))
        entropy = -(p1 * math.log2(p1) + p2 * math.log2(p2))
        margin = abs(p1 - p2)
        return round(entropy, 4), round(margin, 4)

    def detect_faces(self, img_np):

        """
        Safe face bounding box detector using skin mask contours
        with fallback to spatial crops.
        """
        try:
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            # Skin color range in HSV space
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            h, w = img_np.shape[:2]
            min_face_area = (w * h) * 0.04
            face_boxes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_face_area:
                    x, y, fw, fh = cv2.boundingRect(cnt)
                    # Face aspect ratio roughly 0.7 to 1.5
                    aspect = float(fh) / max(1, fw)
                    if 0.6 <= aspect <= 2.0:
                        face_boxes.append((x, y, fw, fh))
            return face_boxes
        except Exception:
            return []

    def load_image(self, image_path):
        """
        Safely load image using PIL with OpenCV fallback for WebP and unsupported PIL formats.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"Could not read image file: {image_path}. File may be corrupted or an unsupported format.")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = self.load_image(image_path)
        img_np = np.array(image)


        # 1. GLOBAL ANALYSIS
        global_real, global_fake = self._analyze_image(image)
        entropy, margin = self.compute_uncertainty(global_real, global_fake)

        # 2. LOCAL ANALYSIS (Face-Aware or Spatial Fallback)
        faces = self.detect_faces(img_np)

        crops = {}
        if len(faces) > 0:
            for idx, (fx, fy, fw, fh) in enumerate(faces):
                pad_w, pad_h = int(fw * 0.15), int(fh * 0.15)
                x1 = max(0, fx - pad_w)
                y1 = max(0, fy - pad_h)
                x2 = min(image.width, fx + fw + pad_w)
                y2 = min(image.height, fy + fh + pad_h)
                crops[f"face_{idx+1}"] = image.crop((x1, y1, x2, y2))
        else:
            # Fallback spatial crops
            width, height = image.size
            crop_w, crop_h = width // 2, height // 2
            crops = {
                "top_left": image.crop((0, 0, crop_w, crop_h)),
                "top_right": image.crop((width - crop_w, 0, width, crop_h)),
                "bottom_left": image.crop((0, height - crop_h, crop_w, height)),
                "bottom_right": image.crop((width - crop_w, height - crop_h, width, height)),
                "center": image.crop((width // 4, height // 4, width - width // 4, height - height // 4)),
            }

        local_results = []
        for name, crop_img in crops.items():
            real_prob, fake_prob = self._analyze_image(crop_img)
            local_results.append({
                "region": name,
                "real_probability": real_prob,
                "fake_probability": fake_prob
            })

        # Find strongest local FAKE evidence
        strongest_fake_crop = max(local_results, key=lambda x: x["fake_probability"])

        return {
            "global_real_probability": round(global_real, 2),
            "global_fake_probability": round(global_fake, 2),
            "local_strongest_region": strongest_fake_crop["region"],
            "local_real_probability": round(strongest_fake_crop["real_probability"], 2),
            "local_fake_probability": round(strongest_fake_crop["fake_probability"], 2),
            "entropy": entropy,
            "margin": margin,
            "face_detected": len(faces) > 0
        }
