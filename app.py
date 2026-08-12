from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import logging

from models.deepfake_detector import DeepfakeDetector
from models.xai_explainer import ViTGradCAM

app = Flask(__name__)

# Load model once at startup
detector = DeepfakeDetector()

UPLOAD_FOLDER = Path("static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "mp4", "avi", "mov", "mp3", "wav", "m4a"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        media_type = request.form.get("media_type", "image")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(UPLOAD_FOLDER / filename)
            return redirect(url_for("result", filename=filename, media_type=media_type))
    return render_template("index.html")


@app.route("/result/<filename>")
def result(filename):

    media_type = request.args.get(
        "media_type",
        "image"
    )

    file_path = UPLOAD_FOLDER / filename

    prediction = "UNKNOWN"
    confidence = 0.0
    risk = "UNKNOWN"
    gradcam_error = False

    # IMAGE
    if media_type == "image":

        result_data = detector.predict(str(file_path))
        
        global_real = result_data["global_real_probability"]
        global_fake = result_data["global_fake_probability"]
        local_real = result_data["local_real_probability"]
        local_fake = result_data["local_fake_probability"]
        local_region = result_data["local_strongest_region"]

        # ---------------------------------------------------------
        # EVIDENCE FUSION
        # ---------------------------------------------------------
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

        # Grad-CAM Generation
        gradcam_filename = f"gradcam_{filename}"
        gradcam_path = UPLOAD_FOLDER / gradcam_filename
        gradcam_error = False
        influential_region = "Unknown"
        gradcam_status = "SUCCESS"
        try:
            # We pass prediction to GradCAM so it knows which class to explain
            explainer = ViTGradCAM(detector)
            _, influential_region = explainer.generate(str(file_path), str(gradcam_path))
        except Exception as e:
            logging.error(f"Grad-CAM generation failed: {e}")
            gradcam_filename = None
            gradcam_error = True
            gradcam_status = "FAILED"
            influential_region = "None"
            
        print("\n" + "=" * 60)
        print("ORIGINX IMAGE ANALYSIS")
        print("=" * 60)
        print(f"File:\n{filename}\n")
        print(f"Global REAL probability:\n{global_real:.2f}%\n")
        print(f"Global FAKE probability:\n{global_fake:.2f}%\n")
        print(f"Global prediction:\n{'FAKE' if global_fake > global_real else 'REAL'}\n")
        print(f"Global confidence:\n{max(global_fake, global_real):.2f}%\n")
        print(f"Risk:\n{risk}\n")
        print(f"Local strongest region:\n{local_region}\n")
        print(f"Local REAL probability:\n{local_real:.2f}%\n")
        print(f"Local FAKE probability:\n{local_fake:.2f}%\n")
        print(f"Final prediction:\n{prediction}\n")
        print(f"Final confidence:\n{confidence:.2f}%\n")
        print(f"Influential region:\n{influential_region}\n")
        print(f"Grad-CAM:\n{gradcam_status}")
        print("=" * 60 + "\n")

    # VIDEO - later
    elif media_type == "video":

        prediction = "VIDEO ANALYSIS"
        confidence = 0
        risk = "PENDING"
        gradcam_filename = filename
        influential_region = "Unknown"

    # AUDIO - later
    elif media_type == "audio":

        prediction = "AUDIO ANALYSIS"
        confidence = 0
        risk = "PENDING"
        gradcam_filename = filename
        influential_region = "Unknown"

    return render_template(
        "result.html",
        filename=filename,
        gradcam_filename=gradcam_filename,
        gradcam_error=gradcam_error,
        influential_region=influential_region,
        media_type=media_type,
        prediction=prediction,
        confidence=confidence,
        risk=risk
    )
if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )