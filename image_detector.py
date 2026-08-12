import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:

    def __init__(self, image_size=224):
        self.image_size = image_size

    def load_image(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Unable to read image.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def resize_image(self, image):

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        return image

    def normalize_image(self, image):

        image = image.astype(np.float32) / 255.0

        return image

    def preprocess(self, image_path):

        image = self.load_image(image_path)

        image = self.resize_image(image)

        image = self.normalize_image(image)

        return image