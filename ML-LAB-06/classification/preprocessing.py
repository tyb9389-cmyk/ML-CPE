import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """Resize one image to img_size x img_size RGB. Returns None if unusable."""

    if image is None or image.size == 0:
        return None

    try:
        # Check image channels & convert to RGB
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return None

        # Resize image using INTER_AREA filter
        image = cv2.resize(
            image,
            (img_size, img_size),
            interpolation=cv2.INTER_AREA
        )
        return image
    except Exception:
        # Catch any unexpected opencv transformation errors
        return None


def to_features(images):
    """(n, h, w, 3) uint8 -> contiguous array for the CNN model."""
    return np.ascontiguousarray(images, dtype=np.uint8)