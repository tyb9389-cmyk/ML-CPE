import cv2
import numpy as np


def preprocess_image(image, img_size=64):

    if image is None or image.size == 0:
        return None

 
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    return image


def to_features(images):
    """Keep pixels as uint8. Standardization happens inside the model
    (Normalization layer), so training and inference always match."""

    return np.ascontiguousarray(images, dtype=np.uint8)
