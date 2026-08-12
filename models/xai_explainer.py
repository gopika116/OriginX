import os
import cv2
import numpy as np
import torch

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class ViTGradCAM:

    def __init__(self, detector):

        self.detector = detector

        self.model = detector.model

        self.processor = detector.processor

        self.device = detector.device

        # Use the final Transformer block
        self.target_layer = self.model.vit.layers[-1]

    def reshape_transform(self, tensor):

        """
        Convert ViT token representation into
        spatial feature representation.
        """

        # Remove CLS token
        tensor = tensor[:, 1:, :]

        # Calculate patch grid
        number_of_patches = tensor.shape[1]

        grid_size = int(
            np.sqrt(number_of_patches)
        )

        # [batch, patches, features]
        # ->
        # [batch, height, width, features]

        tensor = tensor.reshape(
            tensor.shape[0],
            grid_size,
            grid_size,
            tensor.shape[2]
        )

        # ->
        # [batch, features, height, width]

        tensor = tensor.permute(
            0,
            3,
            1,
            2
        )

        return tensor


    def generate(
        self,
        image_path,
        output_path
    ):

        print(
            "Starting Grad-CAM..."
        )


        # ==============================
        # LOAD IMAGE
        # ==============================

        image = Image.open(
            image_path
        ).convert("RGB")


        # Original image for visualization

        original_image = np.array(
            image
        )


        # ==============================
        # PREPARE IMAGE
        # ==============================

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )


        pixel_values = inputs[
            "pixel_values"
        ].to(
            self.device
        )


        # ==============================
        # GET PREDICTED CLASS
        # ==============================

        self.model.eval()


        with torch.no_grad():

            outputs = self.model(
                pixel_values=pixel_values
            )

            predicted_class = torch.argmax(
                outputs.logits,
                dim=1
            ).item()


        print(
            "Grad-CAM target class:",
            predicted_class
        )


        # ==============================
        # GRAD-CAM TARGET
        # ==============================

        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        class HuggingfaceWrapper(torch.nn.Module):
            def __init__(self, model):
                super(HuggingfaceWrapper, self).__init__()
                self.model = model

            def forward(self, x):
                return self.model(pixel_values=x).logits

        wrapped_model = HuggingfaceWrapper(self.model)

        # ==============================
        # CREATE GRAD-CAM
        # ==============================

        cam = GradCAM(
            model=wrapped_model,
            target_layers=[
                self.target_layer
            ],
            reshape_transform=self.reshape_transform
        )


        # ==============================
        # GENERATE CAM
        # ==============================

        grayscale_cam = cam(
            input_tensor=pixel_values,
            targets=targets
        )


        grayscale_cam = grayscale_cam[0]


        # ==============================
        # RESIZE CAM TO ORIGINAL IMAGE
        # ==============================
        
        orig_h, orig_w = original_image.shape[:2]

        grayscale_cam_resized = cv2.resize(
            grayscale_cam,
            (orig_w, orig_h)
        )

        display_image = (
            original_image.astype(
                np.float32
            ) / 255.0
        )

        # ==============================
        # CREATE HEATMAP
        # ==============================

        visualization = show_cam_on_image(
            display_image,
            grayscale_cam_resized,
            use_rgb=True
        )


        # ==============================
        # SAVE RESULT
        # ==============================

        output_folder = os.path.dirname(
            output_path
        )


        if output_folder:

            os.makedirs(
                output_folder,
                exist_ok=True
            )


        cv2.imwrite(
            output_path,
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR
            )
        )

        print("Grad-CAM saved:", output_path)

        # ==============================
        # INFLUENTIAL REGION DETECTION
        # ==============================
        influential_region = "Other image region"

        # 1. Find the single highest activation pixel after slight smoothing to avoid edge artifacts
        smoothed_cam = cv2.GaussianBlur(grayscale_cam_resized, (31, 31), 0)
        max_idx = np.argmax(smoothed_cam)
        cy_orig, cx_orig = np.unravel_index(max_idx, smoothed_cam.shape)

        # 2. Detect face using OpenCV Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray_img = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Find the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            fx, fy, fw, fh = largest_face

            # 3. Check where the centroid falls relative to the face
            if fx <= cx_orig <= fx + fw and fy <= cy_orig <= fy + fh:
                # Inside the face box
                relative_y = (cy_orig - fy) / fh
                if relative_y < 0.33:
                    influential_region = "Eyes / Forehead"
                elif relative_y < 0.66:
                    influential_region = "Nose / Cheeks"
                else:
                    influential_region = "Mouth / Chin"
            else:
                # Check if it's near the face boundary (within 20% of face width/height)
                margin_x = int(fw * 0.2)
                margin_y = int(fh * 0.2)
                if (fx - margin_x) <= cx_orig <= (fx + fw + margin_x) and \
                   (fy - margin_y) <= cy_orig <= (fy + fh + margin_y):
                    influential_region = "Face Boundary"
                else:
                    influential_region = "Background / Other image region"
        else:
            influential_region = "Other image region"

        print("Most influential region:", influential_region)

        return output_path, influential_region