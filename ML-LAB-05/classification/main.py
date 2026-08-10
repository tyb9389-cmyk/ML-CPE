import json
import os

import joblib
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

DATA_PATH = "PetImages"
OUTPUT_DIR = "outputs"
IMG_SIZE = 100
TEST_SIZE = 0.2
MAX_PER_CLASS = 3000   


def main():

    print("=" * 60)
    print("SVM Image Recognition: Cat vs Dog (Lab 5)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

   
    print("\n[Step 2] Preprocessing images...")

    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

   
    print("\n[Step 3] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

   
    kernels = ['linear', 'poly', 'rbf']
    results = {}
    best_acc = 0.0
    best_model = None
    best_scaler = None
    best_kernel_name = ""

    print("\n[Step 4 & 5] Training and Evaluating SVM across 3 Kernels...")

    for kernel in kernels:
        print(f"\n--- Training Kernel: {kernel.upper()} ---")
        
        
        model, scaler = train_svm(X_train, y_train, kernel=kernel)
        
       
        predictions = predict_svm(model, scaler, X_test)
        
        
        cm_path = f"{OUTPUT_DIR}/confusion_matrix_{kernel}.png"
        acc = evaluate_model(y_test, predictions, classes, save_path=cm_path)
        
        results[kernel] = acc
        print(f"Kernel '{kernel}' Accuracy: {acc * 100:.2f}%")

        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_scaler = scaler
            best_kernel_name = kernel

    
    joblib.dump(best_model, f"{OUTPUT_DIR}/svm_model.pkl")  # เซฟชื่อหลักสำหรับ test.py
    joblib.dump(best_model, f"{OUTPUT_DIR}/svm_model_best_{best_kernel_name}.pkl")
    
    print("\n" + "=" * 40)
    print("       SVM KERNEL ACCURACY COMPARISON      ")
    print("=" * 40)
    for k, acc in results.items():
        print(f"- {k.capitalize():<10} Kernel : {acc * 100:.2f}%")
    print("-" * 40)
    print(f"Best Performing Kernel: {best_kernel_name.capitalize()} ({best_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()