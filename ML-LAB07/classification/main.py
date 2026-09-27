import os
import json
import numpy as np
from data_loader import load_data
from preprocessing import preprocess_images
from split_data import split_dataset
from cnn_model import build_cnn_model, train_model
from evaluate import evaluate_model, plot_training_history
from test_cnn import test_random_samples

def main():
    DATA_PATH = "../PetImages" 
    OUTPUT_DIR = "outputs"
    IMG_SIZE = (128, 128)
    MAX_PER_CLASS = 1000 
    
    print("=== [Step 1] Loading Data ===")
    images, labels, classes = load_data(DATA_PATH, max_per_class=MAX_PER_CLASS)
    
    print("=== [Step 2] Preprocessing Images ===")
    X = preprocess_images(images, img_size=IMG_SIZE)
    y = np.array(labels)
    
   
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "classes.json"), "w") as f:
        json.dump(classes, f)
        
    print("=== [Step 3] Splitting Dataset ===")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    print("=== [Step 4] Building & Training CNN Model ===")
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    num_classes = len(classes)
    model = build_cnn_model(input_shape, num_classes)
    model.summary()
    
    model, history = train_model(
        model, X_train, y_train, X_val, y_val, 
        epochs=10, batch_size=32, output_dir=OUTPUT_DIR
    )
    
    print("=== [Step 5] Evaluating Model ===")
    evaluate_model(model, X_test, y_test, classes, output_dir=OUTPUT_DIR)
    plot_training_history(history, output_dir=OUTPUT_DIR)
    
    print("=== [Step 6] Testing Predictions on Samples ===")
    test_random_samples(model, X_test, y_test, classes, output_dir=OUTPUT_DIR)
    
    print("=== Pipeline Complete Successfully! ===")

if __name__ == "__main__":
    main()