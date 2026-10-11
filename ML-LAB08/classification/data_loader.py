import os

import cv2
import numpy as np

from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_folder_data(data_path, img_size=64, max_per_class=None):
    """Load images from  data_path/<class_name>/*.jpg  (one folder per class)."""

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}\n"
            "Expected layout: <data_path>/<class_name>/<image files>"
        )

    images, labels = [], []

    classes = sorted(
        folder for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    )
    print("Detected classes:", classes)

    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        filenames = sorted(
            f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXT)
        )

        loaded = skipped = 0
        for filename in filenames:
            if max_per_class and loaded >= max_per_class:
                break

            image = cv2.imread(os.path.join(class_path, filename))
            image = preprocess_image(image, img_size)

          
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    return np.stack(images), np.array(labels), classes


def load_builtin_data(name="cifar10", max_per_class=None):
    """Load a Keras built-in dataset (downloads automatically on first use).
    The train/test parts are merged so split_data.py can do the splitting."""

    from tensorflow.keras import datasets

    loaders = {
        "cifar10": datasets.cifar10,
        "fashion_mnist": datasets.fashion_mnist,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(loaders)}")

    (x_tr, y_tr), (x_te, y_te) = loaders[name].load_data()
    images = np.concatenate([x_tr, x_te])
    labels = np.concatenate([y_tr, y_te]).ravel()

    # Grayscale (N, H, W) -> (N, H, W, 1)
    if images.ndim == 3:
        images = images[..., np.newaxis]

    if max_per_class:
        keep = np.concatenate([
            np.where(labels == c)[0][:max_per_class] for c in np.unique(labels)
        ])
        images, labels = images[keep], labels[keep]

    if name == "cifar10":
        classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
    else:
        classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    print(f"Loaded {name}: {len(images)} images, {len(classes)} classes")
    return images, labels, classes


def load_data(source, img_size=64, max_per_class=None):
    """source = 'cifar10' | 'fashion_mnist' | path to an image folder."""

    if source in ("cifar10", "fashion_mnist"):
        return load_builtin_data(source, max_per_class)

    return load_folder_data(source, img_size, max_per_class)
