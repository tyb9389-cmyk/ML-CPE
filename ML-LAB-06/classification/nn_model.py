import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def build_model(input_shape=(100, 100, 3), num_classes=2):
    """Build a Simple CNN model with Rescaling layer."""
    model = models.Sequential([
        # Rescale uint8 [0, 255] to float32 [0, 1]
        layers.Rescaling(1.0 / 255, input_shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1 if num_classes == 2 else num_classes, 
                     activation="sigmoid" if num_classes == 2 else "softmax")
    ])

    loss = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"
    
    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=["accuracy"]
    )
    return model


def train_model(X_train, y_train, X_val, y_val, num_classes, output_dir, epochs=30, batch_size=32):
    """Train and save the Neural Network model."""
    input_shape = X_train.shape[1:]
    model = build_model(input_shape=input_shape, num_classes=num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    model_path = os.path.join(output_dir, "nn_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model, history


def predict_model(model, X_test):
    """Generate predictions for test dataset."""
    probabilities = model.predict(X_test)
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)
    return probabilities.argmax(axis=1)