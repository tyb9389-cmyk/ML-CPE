import json
import os

from tensorflow import keras
from tensorflow.keras import layers


CONFIGS = {
    "A_2conv_dense64": {
        "blocks": [(32, 1), (64, 1)],
        "dense": 64,
    },
    "B_3conv_dense128": {
        "blocks": [(32, 1), (64, 1), (128, 1)],
        "dense": 128,
    },
    "C_8conv_dense256": {
        "blocks": [(32, 2), (64, 2), (128, 2), (256, 2)],
        "dense": 256,
    },
}


def count_conv_layers(blocks):
    return sum(n for _, n in blocks)


def vgg_block(model, filters, n_conv):
    """n_conv x (Conv3x3 -> BatchNorm) then halve the resolution."""

    for _ in range(n_conv):
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(2))


def build_model(X_train, num_classes, blocks, dense, lr=1e-3):

    model = keras.Sequential()
    model.add(keras.Input(shape=X_train.shape[1:]))

    
    norm = layers.Normalization(axis=-1)
    norm.adapt(X_train.astype("float32"))
    model.add(norm)

    
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.05))

    for filters, n_conv in blocks:
        vgg_block(model, filters, n_conv)

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                blocks, dense, epochs=10, batch_size=64, verbose=1):
    """Train for exactly `epochs` epochs (no early stopping), so that the
    number of epochs is the only thing that changes in the epoch comparison."""

    model = build_model(X_train, num_classes, blocks, dense)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    return model, history


def predict_model(model, X):

    return model.predict(X, verbose=0).argmax(axis=1)


def save_run(model, history, output_dir, name):
    """Save the model and its training history under output_dir/name."""

    run_dir = os.path.join(output_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    model.save(os.path.join(run_dir, "model.keras"))
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vs]
                   for k, vs in history.history.items()}, f)

    return run_dir
