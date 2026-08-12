from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "Hemg/Deepfake-Detection"

print("Loading OriginX deepfake detection model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

print("OriginX model loaded successfully.")


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    predicted_index = torch.argmax(probabilities).item()

    confidence = (
        probabilities[predicted_index].item()
        * 100
    )

    label = model.config.id2label[predicted_index]

    print("Model output:", label)
    print("Confidence:", confidence)

    # Normalize model labels
    label_lower = label.lower()

    if (
        "fake" in label_lower
        or "deepfake" in label_lower
    ):

        prediction = "FAKE"

    elif (
        "real" in label_lower
        or "realism" in label_lower
    ):

        prediction = "REAL"

    else:

        prediction = label.upper()

    return prediction, round(confidence, 2)