from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import logging
import os

from models.deepfake_detector import DeepfakeDetector
from models.xai_explainer import ViTGradCAM
from models.video_detector import VideoDetector
from models.audio_detector import AudioDetector
from utils.report_generator import ReportGenerator
from originx_video import detect_video

app = Flask(__name__)

# Load model once at startup
detector = DeepfakeDetector()
explainer = ViTGradCAM(detector)
video_detector = VideoDetector(detector, explainer)
audio_detector = AudioDetector()

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


@app.route("/result/<filename>", methods=["GET", "POST"])
def result(filename):
    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file and file.filename != "" and allowed_file(file.filename):
            media_type = request.form.get("media_type", "image")
            new_filename = secure_filename(file.filename)
            file.save(UPLOAD_FOLDER / new_filename)
            return redirect(url_for("result", filename=new_filename, media_type=media_type))
        return redirect(url_for("index"))

    media_type = request.args.get("media_type") or request.form.get("media_type") or "image"
    file_path = UPLOAD_FOLDER / filename


    prediction = "UNKNOWN"
    confidence = 0.0
    risk = "UNKNOWN"
    gradcam_error = False
    gradcam_filename = None
    influential_region = "Unknown"
    detector_scores = None
    video_summary = None
    evidence_list = None
    attention_plot = None


    # IMAGE
    if media_type == "image":
        try:
            result_data = detector.predict(str(file_path))
        except Exception as e:
            logging.error(f"Image analysis failed for {filename}: {e}")
            return render_template("index.html", error="Unable to read or process the image file. Please ensure it is a valid image.")

        global_real = result_data["global_real_probability"]

        global_fake = result_data["global_fake_probability"]
        local_real = result_data["local_real_probability"]
        local_fake = result_data["local_fake_probability"]
        local_region = result_data["local_strongest_region"]

        # EVIDENCE FUSION
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
        try:
            _, influential_region = explainer.generate(str(file_path), str(gradcam_path))
        except Exception as e:
            logging.error(f"Grad-CAM generation failed: {e}")
            gradcam_filename = None
            gradcam_error = True
            influential_region = "None"

    # VIDEO
    elif media_type == "video":

        try:
            video_out_dir = UPLOAD_FOLDER / f"v_out_{filename}"
            video_out_dir.mkdir(parents=True, exist_ok=True)

            v_res = detect_video(str(file_path), output_dir=str(video_out_dir))

            prediction = v_res["verdict"]
            confidence = round(float(v_res["confidence"]) * 100.0, 2)

            fake_prob = float(v_res["fake_probability"])
            if prediction == "FAKE":
                risk = "HIGH RISK" if fake_prob >= 0.70 else "MEDIUM RISK"
            elif prediction == "REAL":
                risk = "LOW RISK"
            else:
                risk = "UNCERTAIN"

            # Parse Grad-CAM frames for visual display
            gradcam_frames = v_res.get("explainability", {}).get("gradcam_frames", [])
            if gradcam_frames and os.path.exists(gradcam_frames[0]):
                rel_path = os.path.relpath(gradcam_frames[0], start=str(UPLOAD_FOLDER))
                gradcam_filename = rel_path.replace("\\", "/")
            else:
                gradcam_filename = None
                gradcam_error = True

            # Parse Attention plot path
            attn_path = v_res.get("explainability", {}).get("attention_plot")
            if attn_path and os.path.exists(attn_path):
                rel_attn = os.path.relpath(attn_path, start=str(UPLOAD_FOLDER))
                attention_plot = rel_attn.replace("\\", "/")


            explanation_data = v_res.get("explanation", {})
            detector_scores = v_res.get("detector_scores", {})
            evidence_list = explanation_data.get("evidence", [])
            video_summary = explanation_data.get("summary", "")
            influential_region = f"Hypothesis: {explanation_data.get('fake_type_hypothesis', 'Deepfake Sequence Anomalies')}"

        except Exception as e:
            logging.error(f"Video analysis failed: {e}")
            prediction = "UNCERTAIN"
            confidence = 0.0
            risk = "ERROR"
            gradcam_error = True

    # AUDIO
    elif media_type == "audio":
        try:
            a_res = audio_detector.analyze_audio(str(file_path))
            prediction = a_res["prediction"]
            confidence = a_res["confidence"]
            risk = a_res["risk"]
            influential_region = a_res["influential_region"]
            gradcam_error = True # No spatial Grad-CAM image for pure audio
        except Exception as e:
            logging.error(f"Audio analysis failed: {e}")
            prediction = "UNCERTAIN"
            confidence = 0.0
            risk = "ERROR"
            gradcam_error = True

    return render_template(
        "result.html",
        filename=filename,
        gradcam_filename=gradcam_filename,
        gradcam_error=gradcam_error,
        influential_region=influential_region,
        media_type=media_type,
        prediction=prediction,
        confidence=confidence,
        risk=risk,
        detector_scores=detector_scores,
        video_summary=video_summary,
        evidence_list=evidence_list,
        attention_plot=attention_plot
    )



@app.route("/download_report/<filename>", methods=["GET", "POST"])
def download_report(filename):
    media_type = request.args.get("media_type") or request.form.get("media_type") or "image"
    prediction = request.args.get("prediction") or request.form.get("prediction") or "UNKNOWN"
    confidence = request.args.get("confidence") or request.form.get("confidence") or "0.0"
    risk = request.args.get("risk") or request.form.get("risk") or "UNKNOWN"
    region = request.args.get("region") or request.form.get("region") or "Unknown"


    report_pdf_name = f"OriginX_Report_{filename}.pdf"
    report_path = UPLOAD_FOLDER / report_pdf_name

    final_path = ReportGenerator.generate_pdf_report(
        filename=filename,
        media_type=media_type,
        prediction=prediction,
        confidence=confidence,
        risk=risk,
        influential_region=region,
        output_path=str(report_path)
    )

    return send_file(final_path, as_attachment=True)


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )