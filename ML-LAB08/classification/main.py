import csv
import json
import os
import time

import numpy as np
from tensorflow import keras

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from dcnn_model import (CONFIGS, count_conv_layers, train_model,
                        predict_model, save_run)
from evaluate import (evaluate_model, plot_history,
                      plot_config_comparison, plot_epoch_comparison)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


DATASET = "cifar10"
IMG_SIZE = 64          
MAX_PER_CLASS = 2000   
TEST_SIZE = 0.2
VAL_SIZE = 0.1
BATCH_SIZE = 64


CONFIG_EPOCHS = 10


EPOCH_LIST = [5, 10, 20]
EPOCH_CONFIG = "B_3conv_dense128"
# ------------------------------------------


def run_experiment(name, cfg, epochs, data, num_classes, verbose=1):

    X_train, X_val, X_test, y_train, y_val, y_test = data

    start = time.time()
    model, history = train_model(
        X_train, y_train, X_val, y_val, num_classes,
        cfg["blocks"], cfg["dense"], epochs, BATCH_SIZE, verbose
    )
    seconds = time.time() - start

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    val_acc = history.history["val_accuracy"][-1]

    result = {
        "config": name,
        "conv_layers": count_conv_layers(cfg["blocks"]),
        "dense_neurons": cfg["dense"],
        "epochs": epochs,
        "train_accuracy": history.history["accuracy"][-1],
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "seconds": round(seconds, 1),
    }
    print(f"  -> {name} | {epochs} epochs | test acc {test_acc * 100:.2f}% "
          f"| {seconds:.0f}s")

    return model, history, result


def save_results(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {path}")


def print_table(results):
    header = (f"{'Config':<20}{'Conv':>5}{'Dense':>7}{'Epochs':>8}"
              f"{'Train':>9}{'Val':>9}{'Test':>9}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['config']:<20}{r['conv_layers']:>5}{r['dense_neurons']:>7}"
              f"{r['epochs']:>8}{r['train_accuracy'] * 100:>8.2f}%"
              f"{r['val_accuracy'] * 100:>8.2f}%{r['test_accuracy'] * 100:>8.2f}%")


def main():

    print("--" * 30)
    print(f"DCNN Lab: dataset = {DATASET}")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    keras.utils.set_random_seed(42)

    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATASET, IMG_SIZE, MAX_PER_CLASS)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)
    print(f"Total images: {len(images)} | Classes: {classes}")

    
    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

  
    print("\n[Step 3] Splitting dataset...")
    data = split_dataset(X, y, TEST_SIZE, VAL_SIZE)
    X_train, X_val, X_test, y_train, y_val, y_test = data
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)
    print(f"Train: {len(X_train)} | Validation: {len(X_val)} | "
          f"Test: {len(X_test)}")

    num_classes = len(classes)

   
    print(f"\n[Step 4] Experiment 1: configurations "
          f"({CONFIG_EPOCHS} epochs each)")
    config_results = []
    best = {"acc": -1, "model": None}

    for name, cfg in CONFIGS.items():
        print(f"\nTraining {name} ...")
        model, history, result = run_experiment(
            name, cfg, CONFIG_EPOCHS, data, num_classes
        )
        config_results.append(result)

        run_dir = save_run(model, history, OUTPUT_DIR,
                           f"config_{name}_ep{CONFIG_EPOCHS}")
        plot_history(history, os.path.join(run_dir, "training_history.png"),
                     f"{name} ({CONFIG_EPOCHS} epochs)")

        if result["test_accuracy"] > best["acc"]:
            best = {"acc": result["test_accuracy"], "model": model,
                    "name": name}

    save_results(config_results, f"{OUTPUT_DIR}/results_configs.csv")
    plot_config_comparison(config_results, f"{OUTPUT_DIR}/compare_configs.png")

    
    print(f"\n[Step 5] Experiment 2: epochs {EPOCH_LIST} "
          f"with {EPOCH_CONFIG}")
    epoch_results = []
    epoch_best = {"acc": -1, "model": None}

    for epochs in EPOCH_LIST:
        print(f"\nTraining {EPOCH_CONFIG} for {epochs} epochs ...")
        model, history, result = run_experiment(
            EPOCH_CONFIG, CONFIGS[EPOCH_CONFIG], epochs, data, num_classes
        )
        epoch_results.append(result)

        run_dir = save_run(model, history, OUTPUT_DIR,
                           f"epochs_{EPOCH_CONFIG}_ep{epochs}")
        plot_history(history, os.path.join(run_dir, "training_history.png"),
                     f"{EPOCH_CONFIG} ({epochs} epochs)")

        if result["test_accuracy"] > epoch_best["acc"]:
            epoch_best = {"acc": result["test_accuracy"], "model": model,
                          "name": f"{EPOCH_CONFIG} / {epochs} epochs"}

    save_results(epoch_results, f"{OUTPUT_DIR}/results_epochs.csv")
    plot_epoch_comparison(epoch_results, f"{OUTPUT_DIR}/compare_epochs.png",
                          EPOCH_CONFIG)

    
    print("\n[Step 6] Summary")
    print("\nExperiment 1: DCNN configurations")
    print_table(config_results)
    print(f"\nExperiment 2: epochs ({EPOCH_CONFIG})")
    print_table(epoch_results)

    overall = max([best, epoch_best], key=lambda b: b["acc"])
    print(f"\nBest model overall: {overall['name']} "
          f"({overall['acc'] * 100:.2f}% test accuracy)")

    predictions = predict_model(overall["model"], X_test)
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    overall["model"].save(f"{OUTPUT_DIR}/best_model.keras")
    print(f"Saved: {OUTPUT_DIR}/best_model.keras")


if __name__ == "__main__":
    main()
