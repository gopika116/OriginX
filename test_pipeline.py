import os
from PIL import Image
from models.deepfake_detector import DeepfakeDetector
from models.xai_explainer import ViTGradCAM
import logging

logging.basicConfig(level=logging.INFO)

def create_dummy_image(path, size, color):
    img = Image.new('RGB', size, color=color)
    img.save(path)

def main():
    print("=" * 60)
    print("ORIGINX AI - COMPREHENSIVE PIPELINE TEST")
    print("=" * 60)

    # Prepare test directories
    os.makedirs("tests", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)

    test_images = [
        {"name": "real_mock.jpg", "size": (256, 256), "color": "blue"},
        {"name": "fake_mock.jpg", "size": (512, 512), "color": "red"},
        {"name": "compressed_mock.jpg", "size": (256, 256), "color": "green", "quality": 10},
        {"name": "large_mock.png", "size": (1024, 1024), "color": "yellow"}
    ]

    for img_data in test_images:
        path = os.path.join("tests", img_data["name"])
        if not os.path.exists(path):
            img = Image.new('RGB', img_data["size"], color=img_data["color"])
            if "quality" in img_data:
                img.save(path, "JPEG", quality=img_data["quality"])
            else:
                img.save(path)

    detector = DeepfakeDetector()
    explainer = ViTGradCAM(detector)

    print("\nRunning tests...\n")

    for img_data in test_images:
        image_path = os.path.join("tests", img_data["name"])
        gradcam_path = os.path.join("static", "uploads", f"gradcam_{img_data['name']}")
        
        print("-" * 50)
        print(f"Testing image: {image_path}")

        # Test Prediction
        try:
            result = detector.predict(image_path)
            global_real = result["global_real_probability"]
            global_fake = result["global_fake_probability"]
            local_real = result["local_real_probability"]
            local_fake = result["local_fake_probability"]
            local_region = result["local_strongest_region"]

            if global_fake >= 80 or (global_fake >= 60 and local_fake >= 80):
                prediction = "FAKE"
                risk = "HIGH RISK"
                confidence = max(global_fake, local_fake)
            elif global_fake >= 60 and local_fake < 60:
                prediction = "FAKE"
                risk = "MEDIUM RISK"
                confidence = global_fake
            elif global_real >= 80 and local_fake < 60:
                prediction = "REAL"
                risk = "LOW RISK"
                confidence = global_real
            elif global_real >= 60 and local_fake < 40:
                prediction = "REAL"
                risk = "LOW RISK"
                confidence = global_real
            else:
                prediction = "UNCERTAIN"
                risk = "UNCERTAIN"
                confidence = max(global_real, global_fake)

            # Test Grad-CAM
            try:
                _, influential_region = explainer.generate(image_path, gradcam_path)
                gradcam_status = "SUCCESS"
            except Exception as e:
                gradcam_status = f"FAILED: {e}"
                influential_region = "None"

            print("\n" + "=" * 40)
            print("ORIGINX IMAGE ANALYSIS")
            print("=" * 40)
            print(f"Global REAL:\n{global_real:.2f}%")
            print(f"Global FAKE:\n{global_fake:.2f}%\n")
            print(f"Strongest Local Crop:\n{local_region}")
            print(f"Local REAL:\n{local_real:.2f}%")
            print(f"Local FAKE:\n{local_fake:.2f}%\n")
            print(f"Final Prediction:\n{prediction}")
            print(f"Final Confidence:\n{confidence:.2f}%")
            print(f"Risk Level:\n{risk}\n")
            print(f"Most Influential Region:\n{influential_region}\n")
            print(f"Grad-CAM:\n{gradcam_status}")
            print("=" * 40 + "\n")

        except Exception as e:
            print(f"Prediction failed: {e}")
            continue

if __name__ == "__main__":
    main()
