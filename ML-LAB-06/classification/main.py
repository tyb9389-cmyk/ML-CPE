import json
import os

import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from nn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history
from test_nn import run_test_sample  # นำเข้าฟังก์ชันทดสอบสุ่มภาพ

# Paths are relative to this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "PetImages")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 3000   # None = use all images
EPOCHS = 30
BATCH_SIZE = 32


def main():
    print("=" * 60)
    print("Neural Network Image Recognition: Cat vs Dog")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    if len(images) == 0:
        print("Error: No images found. Please check your data directory.")
        return

    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
    with open(os.path.join(OUTPUT_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=4)

    print(f"Total images loaded : {len(images)}")
    print(f"Detected Classes    : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels

    np.save(os.path.join(OUTPUT_DIR, "features.npy"), X)
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")
    print(f"Testing samples    : {len(X_test)}")

    # Step 4: Train Model
    print("\n[Step 4] Training model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )

    # Save training history as JSON
    hist_dict = history.history if hasattr(history, "history") else history
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(hist_dict, f, indent=4)

    print("Training completed & history saved.")

    # Step 5: Prediction & Evaluation
    print("\n[Step 5] Evaluating model on Test set...")
    predictions = predict_model(model, X_test)

    evaluate_model(
        y_test, predictions, classes,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    plot_history(history, os.path.join(OUTPUT_DIR, "training_history.png"))

    # Step 6: Visual Inference Test
    print("\n[Step 6] Running random test inference...")
    test_sample_path = os.path.join(OUTPUT_DIR, "prediction_sample.png")
    run_test_sample(model, X_test, y_test, classes, save_path=test_sample_path)

    print("\n" + "=" * 60)
    print(f"Pipeline executed successfully! Artifacts saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()