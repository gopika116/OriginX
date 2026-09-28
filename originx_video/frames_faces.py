"""Frame extraction and face detection module for OriginX_Video.

Sequential-decode stride sampling (16 frames) with OpenCV DNN Caffe face detector
and automatic Haar cascade fallback. Extracts square face crops with 25% margin.
"""

import os
import sys
import logging
import urllib.request
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

# Configure logger
logger = logging.getLogger("originx_video.frames_faces")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Caffe model URLs and fallback mirrors
CAFFE_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "caffe")
)


class FaceDetector:
    """OpenCV DNN Caffe face detector with automatic weights download and Haar fallback."""

    def __init__(self, models_dir: Optional[str] = None, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.net = None
        self.haar_cascade = None
        self.using_dnn = False
        self._init_detector()

    def _ensure_caffe_models(self) -> Tuple[str, str]:
        os.makedirs(self.models_dir, exist_ok=True)
        prototxt_path = os.path.join(self.models_dir, "deploy.prototxt")
        caffemodel_path = os.path.join(self.models_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        if not os.path.exists(prototxt_path) or os.path.getsize(prototxt_path) < 100:
            logger.info(f"Downloading Caffe prototxt to {prototxt_path}...")
            urllib.request.urlretrieve(CAFFE_PROTOTXT_URL, prototxt_path)

        if not os.path.exists(caffemodel_path) or os.path.getsize(caffemodel_path) < 1000000:
            logger.info(f"Downloading Caffe model weights to {caffemodel_path} (approx 10MB)...")
            urllib.request.urlretrieve(CAFFE_MODEL_URL, caffemodel_path)

        return prototxt_path, caffemodel_path

    def _init_detector(self):
        try:
            prototxt_path, caffemodel_path = self._ensure_caffe_models()
            if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
                if hasattr(cv2.dnn, 'readNetFromCaffe'):
                    self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                elif hasattr(cv2.dnn, 'readNet'):
                    self.net = cv2.dnn.readNet(prototxt_path, caffemodel_path)
                
                if self.net is not None:
                    self.using_dnn = True
                    logger.info("OpenCV DNN Caffe face detector successfully initialized.")
                    return
        except Exception as e:
            logger.warning(f"Could not load Caffe face detector ({e}). Falling back to Haar/HSV cascade.")

        # Fallback to Haar Cascade
        try:
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(haar_path) and hasattr(cv2, 'CascadeClassifier'):
                self.haar_cascade = cv2.CascadeClassifier(haar_path)
                self.using_dnn = False
                logger.info("OpenCV Haar Cascade face detector successfully initialized.")
                return
        except Exception:
            pass

        logger.info("Using HSV skin-contour face detection fallback.")

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect all faces in frame. Returns list of (x, y, w, h) bounding boxes."""
        h, w = frame_bgr.shape[:2]
        boxes = []

        if self.using_dnn and self.net is not None:
            try:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame_bgr, (300, 300)),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0),
                    swapRB=False,
                    crop=False,
                )
                self.net.setInput(blob)
                detections = self.net.forward()
                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence >= self.conf_threshold:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype("int")
                        bw = max(0, x2 - x1)
                        bh = max(0, y2 - y1)
                        if bw > 10 and bh > 10:
                            boxes.append((max(0, x1), max(0, y1), bw, bh))
                if boxes:
                    return boxes
            except Exception as e:
                logger.warning(f"DNN detection failed on frame: {e}. Trying fallback.")

        # Haar cascade fallback
        if self.haar_cascade is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                detected = self.haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )
                for (x, y, bw, bh) in detected:
                    boxes.append((int(x), int(y), int(bw), int(bh)))
                if boxes:
                    return boxes
            except Exception:
                pass

        # HSV Skin contour fallback
        try:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_face_area = (w * h) * 0.03
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_face_area:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect = float(bh) / max(1, bw)
                    if 0.6 <= aspect <= 2.0:
                        boxes.append((int(bx), int(by), int(bw), int(bh)))
        except Exception:
            pass

        return boxes



def crop_face_square(
    frame_bgr: np.ndarray,
    box: Optional[Tuple[int, int, int, int]],
    margin: float = 0.25,
    min_size: int = 48,
    target_size: Optional[int] = 224,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract square face crop with margin. If box is None, returns center square crop."""
    h, w = frame_bgr.shape[:2]

    if box is not None:
        bx, by, bw, bh = box
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        # Margin 25% means side length is max(bw, bh) * (1 + 2 * margin) = max(bw, bh) * 1.5
        side = max(bw, bh) * (1.0 + 2.0 * margin)
    else:
        # Fallback to center square
        cx = w / 2.0
        cy = h / 2.0
        side = min(w, h) * 0.8

    side = max(side, min_size)
    half = side / 2.0

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + int(round(side))
    y2 = y1 + int(round(side))

    # Pad if outside boundaries to guarantee exact square shape
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w, x2)
    crop_y2 = min(h, y2)

    cropped = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

    # Ensure square dimensions
    ch, cw = cropped.shape[:2]
    if ch != cw:
        final_side = min(ch, cw)
        cropped = cropped[:final_side, :final_side]

    # Resize if smaller than min_size
    if cropped.shape[0] < min_size or cropped.shape[1] < min_size:
        cropped = cv2.resize(cropped, (min_size, min_size), interpolation=cv2.INTER_LINEAR)

    if target_size is not None and (cropped.shape[0] != target_size or cropped.shape[1] != target_size):
        cropped_resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return cropped_resized, (x1, y1, x2 - x1, y2 - y1)

    return cropped, (x1, y1, x2 - x1, y2 - y1)


def sample_video_frames(
    video_path: str,
    num_frames: int = 16,
) -> Tuple[List[np.ndarray], List[int], float]:
    """Sample exactly num_frames using sequential decode stride sampling (no random seeks)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sequential scan to determine frame indices
    if total_frames <= 0:
        # Unknown total frames: decode all into memory/list if short
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        total_frames = len(all_frames)
        if total_frames == 0:
            raise ValueError(f"Video file contains 0 decodable frames: {video_path}")
        target_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        sampled = [all_frames[idx] for idx in target_indices]
        return sampled, target_indices, fps

    # Target indices spread evenly across the video duration
    if total_frames < num_frames:
        target_indices = list(range(total_frames))
        # Will pad after reading
    else:
        target_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    target_set = set(target_indices)
    frames_dict: Dict[int, np.ndarray] = {}

    cur_idx = 0
    max_target = max(target_indices)

    while cur_idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            break
        if cur_idx in target_set:
            frames_dict[cur_idx] = frame
        cur_idx += 1

    cap.release()

    # Reconstruct in order of target_indices
    sampled_frames = []
    last_valid = None
    for idx in target_indices:
        if idx in frames_dict:
            last_valid = frames_dict[idx]
            sampled_frames.append(last_valid)
        elif last_valid is not None:
            sampled_frames.append(last_valid.copy())

    # Fallback padding if video ended prematurely
    while len(sampled_frames) < num_frames:
        if sampled_frames:
            sampled_frames.append(sampled_frames[-1].copy())
        else:
            raise ValueError(f"Unable to decode frames from: {video_path}")

    return sampled_frames[:num_frames], target_indices[:num_frames], fps


def extract_frames_and_faces(
    video_path: str,
    num_frames: int = 16,
    margin: float = 0.25,
    min_size: int = 48,
    target_size: Optional[int] = 224,
    detector: Optional[FaceDetector] = None,
) -> Dict[str, Any]:
    """Main extraction routine: samples 16 frames, detects largest face per frame,

    returns square crops, face masks, counts, and metadata.
    """
    if detector is None:
        detector = FaceDetector()

    raw_frames, target_indices, fps = sample_video_frames(video_path, num_frames=num_frames)

    face_crops = []
    face_masks = []
    face_boxes = []
    faces_found_per_frame = []

    for i, frame in enumerate(raw_frames):
        detected_boxes = detector.detect_faces(frame)
        faces_count = len(detected_boxes)
        faces_found_per_frame.append(faces_count)

        logger.info(f"Frame {i} (index {target_indices[i]}): detected {faces_count} face(s)")

        if faces_count > 0:
            # Pick largest face by area (w * h)
            largest_box = max(detected_boxes, key=lambda b: b[2] * b[3])
            crop, crop_box = crop_face_square(
                frame,
                box=largest_box,
                margin=margin,
                min_size=min_size,
                target_size=target_size,
            )
            face_crops.append(crop)
            face_masks.append(True)
            face_boxes.append(crop_box)
        else:
            # No face detected in this frame: masked handling
            logger.warning(f"Frame {i}: No face detected! Applying center crop with face_mask=False.")
            crop, crop_box = crop_face_square(
                frame,
                box=None,
                margin=margin,
                min_size=min_size,
                target_size=target_size,
            )
            face_crops.append(crop)
            face_masks.append(False)
            face_boxes.append(None)

    total_faces_detected = sum(1 for m in face_masks if m)

    return {
        "raw_frames": raw_frames,
        "face_crops": face_crops,
        "face_masks": face_masks,
        "face_boxes": face_boxes,
        "faces_found_per_frame": faces_found_per_frame,
        "total_faces_detected": total_faces_detected,
        "frame_indices": target_indices,
        "fps": fps,
        "video_path": video_path,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python frames_faces.py <video_path>")
        sys.exit(1)

    video_input = sys.argv[1]
    print(f"Testing extraction on {video_input}...")
    res = extract_frames_and_faces(video_input)
    print(f"Sampled {len(res['raw_frames'])} frames.")
    print(f"Faces found per frame: {res['faces_found_per_frame']}")
    print(f"Total frames with face: {res['total_faces_detected']}/{len(res['face_crops'])}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "self_check_p1_output")
    os.makedirs(out_dir, exist_ok=True)

    for idx, (frame, crop, mask) in enumerate(zip(res["raw_frames"], res["face_crops"], res["face_masks"])):
        fpath = os.path.join(out_dir, f"frame_{idx:02d}.jpg")
        cpath = os.path.join(out_dir, f"crop_{idx:02d}.jpg")
        cv2.imwrite(fpath, frame)
        cv2.imwrite(cpath, crop)
        assert crop.shape[0] == crop.shape[1], f"Crop {idx} not square: {crop.shape}"
        assert crop.shape[0] >= 48, f"Crop {idx} smaller than 48px: {crop.shape}"
        print(f"Frame {idx:02d}: path={fpath}, crop={cpath}, size={crop.shape}, face_detected={mask}")

    print("Self-check P1 PASSED!")
