import os
import cv2
import numpy as np
from PIL import Image

class VideoDetector:

    def __init__(self, detector, explainer=None):
        self.detector = detector
        self.explainer = explainer

    def analyze_video(self, video_path, output_folder="static/uploads", sample_count=10):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = total_frames / fps if total_frames > 0 else 0

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains no readable frames.")

        # Determine frame sample indices uniformly across video
        step = max(1, total_frames // sample_count)
        sampled_indices = list(range(0, total_frames, step))[:sample_count]

        frame_results = []
        highest_fake_score = -1.0
        peak_frame_img = None
        peak_frame_index = 0

        for frame_idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Save temporary frame image for analysis
            temp_frame_path = os.path.join(output_folder, f"temp_frame_{frame_idx}.jpg")
            pil_img.save(temp_frame_path)

            try:
                res = self.detector.predict(temp_frame_path)
                real_p = res["global_real_probability"]
                fake_p = res["global_fake_probability"]

                frame_results.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 2),
                    "real_probability": real_p,
                    "fake_probability": fake_p,
                })

                if fake_p > highest_fake_score:
                    highest_fake_score = fake_p
                    peak_frame_img = temp_frame_path
                    peak_frame_index = frame_idx
                else:
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)

            except Exception as e:
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)

        cap.release()

        if not frame_results:
            raise ValueError("Failed to analyze any video frames.")

        # Aggregation Logic
        avg_real = np.mean([f["real_probability"] for f in frame_results])
        avg_fake = np.mean([f["fake_probability"] for f in frame_results])
        max_fake = max([f["fake_probability"] for f in frame_results])

        # Evidence Decision Logic for Video
        if max_fake >= 80.0 or avg_fake >= 65.0:
            prediction = "FAKE"
            risk = "HIGH RISK" if max_fake >= 85.0 else "MEDIUM RISK"
            confidence = round(max(avg_fake, max_fake), 2)
        elif avg_real >= 70.0 and max_fake < 45.0:
            prediction = "REAL"
            risk = "LOW RISK"
            confidence = round(avg_real, 2)
        else:
            prediction = "UNCERTAIN"
            risk = "UNCERTAIN"
            confidence = round(max(avg_real, avg_fake), 2)

        # Generate Grad-CAM on peak suspicious frame
        gradcam_filename = None
        influential_region = "Multiple video frames"
        if peak_frame_img and self.explainer:
            try:
                base_name = os.path.basename(video_path)
                gradcam_filename = f"gradcam_video_{base_name}.jpg"
                gradcam_out_path = os.path.join(output_folder, gradcam_filename)
                _, influential_region = self.explainer.generate(peak_frame_img, gradcam_out_path)
            except Exception as e:
                gradcam_filename = None
            finally:
                if os.path.exists(peak_frame_img):
                    os.remove(peak_frame_img)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk": risk,
            "avg_real_probability": round(float(avg_real), 2),
            "avg_fake_probability": round(float(avg_fake), 2),
            "max_fake_probability": round(float(max_fake), 2),
            "total_frames_analyzed": len(frame_results),
            "total_video_frames": total_frames,
            "duration_seconds": round(float(duration), 2),
            "fps": round(float(fps), 2),
            "peak_frame_index": peak_frame_index,
            "gradcam_filename": gradcam_filename,
            "influential_region": influential_region
        }
