import os
import cv2
import numpy as np

from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_data(data_path, img_size=100, max_per_class=None):
    images = []
    labels = []

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path not found: {data_path}")

    # Detect classes automatically from subdirectories
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])
    print("Detected classes:", classes)

    # Read images from each class directory
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0
        for filename in filenames:
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            # Skip immediately if image failed to load
            if image is None:
                skipped += 1
                continue

            # Process valid image
            processed_img = preprocess_image(image, img_size)

            if processed_img is None:
                skipped += 1
                continue

            images.append(processed_img)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    if len(images) == 0:
        print("Warning: No images loaded!")
        return np.array([]), np.array([]), classes

    return np.array(images), np.array(labels), classes