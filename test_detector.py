from models.deepfake_detector import DeepfakeDetector
import os


print("=" * 50)
print("ORIGINX AI - DEEPFAKE DETECTOR TEST")
print("=" * 50)


image_path = input(
    "Enter the path of an image to test: "
).strip()


if not os.path.exists(image_path):

    print()
    print("ERROR: Image file not found.")
    print("Please check the image path.")
    exit()


print()
print("Loading AI model...")
print()


try:

    detector = DeepfakeDetector()

    print()
    print("Analyzing image...")
    print()

    result = detector.predict(
        image_path
    )


    print("=" * 50)

    print("DETECTION RESULT")

    print("=" * 50)

    print(
        "Prediction       :",
        result["prediction"]
    )

    print(
        "Confidence       :",
        result["confidence"],
        "%"
    )

    print(
        "Real Probability :",
        result["real_probability"],
        "%"
    )

    print(
        "Fake Probability :",
        result["fake_probability"],
        "%"
    )

    print("=" * 50)


except Exception as error:

    print()
    print("ERROR DURING DETECTION")
    print()
    print(error)