import os

from models.deepfake_detector import DeepfakeDetector
from models.xai_explainer import ViTGradCAM


print("=" * 60)
print("ORIGINX AI - EXPLAINABLE AI TEST")
print("=" * 60)


image_path = input(
    "Enter image path: "
).strip()


if not os.path.exists(image_path):

    print("Image not found.")

    exit()


print()
print("Loading deepfake detector...")
print()

detector = DeepfakeDetector()


print()
print("Running prediction...")
print()

result = detector.predict(
    image_path
)


print("=" * 60)
print("PREDICTION")
print("=" * 60)

print(
    "Prediction:",
    result["prediction"]
)

print(
    "Confidence:",
    result["confidence"],
    "%"
)


print()
print("Generating explanation...")
print()


explainer = ViTGradCAM(
    detector
)


output_path = (
    "uploads/explanation.jpg"
)


try:

    explainer.generate(
        image_path,
        output_path
    )

    print()
    print("=" * 60)
    print("XAI COMPLETE")
    print("=" * 60)

    print(
        "Explanation saved to:"
    )

    print(
        output_path
    )


except Exception as error:

    print()
    print("=" * 60)
    print("XAI ERROR")
    print("=" * 60)

    print(error)