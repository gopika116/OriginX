import os
import torch

from PIL import Image
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


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

        print("Deepfake detection model loaded successfully.")
        print(f"Model architecture: {type(self.model).__name__}")
        print(f"Label mapping: {self.model.config.id2label}")


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

    def predict(self, image_path):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # 1. GLOBAL ANALYSIS
        global_real, global_fake = self._analyze_image(image)

        # 2. LOCAL ANALYSIS (Overlapping Crops)
        width, height = image.size
        # 50% crops
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
            "local_fake_probability": round(strongest_fake_crop["fake_probability"], 2)
        }