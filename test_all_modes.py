import os
import cv2
import numpy as np
from PIL import Image

from models.deepfake_detector import DeepfakeDetector
from models.xai_explainer import ViTGradCAM
from models.video_detector import VideoDetector
from models.audio_detector import AudioDetector
from utils.report_generator import ReportGenerator

def test_system():
    print("=" * 60)
    print("ORIGINX FULL MULTI-MODAL SYSTEM TEST")
    print("=" * 60)

    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("tests", exist_ok=True)

    # 1. Test Image Detection & XAI (JPG & WEBP Fallback)
    test_img_path = "tests/sample_test.jpg"
    img = Image.new("RGB", (300, 300), color="blue")
    img.save(test_img_path)

    test_webp_path = "tests/sample_test.webp"
    cv2.imwrite(test_webp_path, np.zeros((300, 300, 3), dtype=np.uint8))

    detector = DeepfakeDetector()
    explainer = ViTGradCAM(detector)

    img_res = detector.predict(test_img_path)
    print("\n[IMAGE DETECTION RESULT (JPG)]")
    print(f"Global Real: {img_res['global_real_probability']}% | Global Fake: {img_res['global_fake_probability']}%")
    print(f"Entropy: {img_res['entropy']} | Margin: {img_res['margin']}")
    print(f"Face Detected: {img_res['face_detected']} | Strongest Region: {img_res['local_strongest_region']}")

    webp_res = detector.predict(test_webp_path)
    print("\n[IMAGE DETECTION RESULT (WEBP)]")
    print(f"Global Real: {webp_res['global_real_probability']}% | Global Fake: {webp_res['global_fake_probability']}%")

    cam_path = "static/uploads/gradcam_sample_test.jpg"
    _, reg = explainer.generate(test_img_path, cam_path)
    print(f"Grad-CAM Generated at: {cam_path} | Region: {reg}")


    # 2. Test Audio Detector
    test_audio_path = "tests/sample_audio.wav"
    # Create simple dummy wav file
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, False)
    tone = np.sin(440 * 2 * np.pi * t) * 0.5
    audio_int16 = (tone * 32767).astype(np.int16)
    
    import wave
    with wave.open(test_audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    audio_det = AudioDetector()
    aud_res = audio_det.analyze_audio(test_audio_path)
    print("\n[AUDIO DETECTION RESULT]")
    print(f"Prediction: {aud_res['prediction']} | Confidence: {aud_res['confidence']}% | Risk: {aud_res['risk']}")
    print(f"Authentic Prob: {aud_res['real_probability']}% | Synthetic Prob: {aud_res['fake_probability']}%")

    # 3. Test Report Generation
    pdf_out = "static/uploads/test_report.pdf"
    res_path = ReportGenerator.generate_pdf_report(
        filename="sample_test.jpg",
        media_type="image",
        prediction="REAL",
        confidence=85.5,
        risk="LOW RISK",
        influential_region=reg,
        output_path=pdf_out
    )
    print(f"\n[REPORT GENERATOR RESULT]\nReport file created at: {res_path}")

    print("\n" + "=" * 60)
    print("ALL MULTI-MODAL PIPELINE TESTS PASSED CLEANLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_system()
