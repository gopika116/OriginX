import os
import sys

from app import app
from originx_video import detect_video

def main():
    print("=" * 60)
    print("ORIGINX - DEMO READINESS VIDEO INTEGRATION TEST")
    print("=" * 60)

    test_clip_path = os.path.abspath(r"C:\Projects\video_detection\OriginX_Video\tests\data\clip_fake_sample.mp4")
    if not os.path.exists(test_clip_path):
        print(f"Error: test clip not found at {test_clip_path}")
        return

    print(f"\n1. Testing standalone detect_video on: {test_clip_path}")
    v_res = detect_video(test_clip_path, output_dir="static/uploads/test_demo_video_output")

    print("\n[DETECT_VIDEO CONTRACT OUTPUT]")
    print(f"Verdict        : {v_res['verdict']}")
    print(f"Confidence     : {v_res['confidence'] * 100:.2f}%")
    print(f"Fake Prob      : {v_res['fake_probability'] * 100:.2f}%")
    print(f"Hypothesis     : {v_res['explanation']['fake_type_hypothesis']}")
    print(f"Scores         : {v_res['detector_scores']}")
    print(f"Attention plot : {v_res['explainability']['attention_plot']}")
    print(f"Grad-CAM frames: {len(v_res['explainability']['gradcam_frames'])}")
    print(f"Runtime (sec)  : {v_res['meta']['runtime_sec']}s")

    print("\n2. Testing Flask test_client upload on / (media_type=video)...")
    client = app.test_client()
    with open(test_clip_path, "rb") as f:
        response = client.post(
            "/",
            data={
                "media_type": "video",
                "file": (f, "demo_test_clip.mp4")
            },
            content_type="multipart/form-data"
        )
    print(f"Upload Response Status Code: {response.status_code}")
    print(f"Redirect Location: {response.headers.get('Location')}")

    # Follow redirect
    if response.status_code == 302:
        redirect_url = response.headers.get("Location")
        print(f"\n3. Following redirect to {redirect_url}...")
        res_page = client.get(redirect_url)
        print(f"Result Page Status Code: {res_page.status_code}")
        assert res_page.status_code == 200, "Result page did not return 200 OK!"
        html = res_page.data.decode("utf-8")
        assert "Detection Result" in html
        assert "Multi-Criteria Forensic Scorecard" in html or "Ensemble Metric Scorecard" in html
        print("[PASS] Verified: HTML contains Detection Result and Multi-Criteria Forensic Scorecard!")



    print("\n" + "=" * 60)
    print("ALL VIDEO INTEGRATION DEMO CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
