import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(y_test, predictions, classes, save_path=None, verbose=True):

    labels = list(range(len(classes)))
    accuracy = accuracy_score(y_test, predictions)

    if verbose:
        print("\n------------ Evaluation ------------------")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(
            y_test, predictions, labels=labels,
            target_names=[str(c) for c in classes], zero_division=0
        ))

    if save_path:
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    size = max(5, 0.6 * len(classes))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, save_path, title=""):
    """Training vs validation accuracy / loss."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, key, label in ((axes[0], "accuracy", "Accuracy"),
                           (axes[1], "loss", "Loss")):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history["val_" + key], label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_config_comparison(results, save_path):
    """Bar chart: test accuracy of each DCNN configuration."""

    names = [r["config"] for r in results]
    acc = [r["test_accuracy"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, acc)
    ax.bar_label(bars, fmt="%.2f%%")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("DCNN configuration comparison")
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_epoch_comparison(results, save_path, config_name=""):
    """Line chart: test accuracy vs number of epochs."""

    epochs = [r["epochs"] for r in results]
    test_acc = [r["test_accuracy"] * 100 for r in results]
    val_acc = [r["val_accuracy"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, test_acc, marker="o", label="test")
    ax.plot(epochs, val_acc, marker="s", label="validation")
    for x, y in zip(epochs, test_acc):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center")
    ax.set_xticks(epochs)
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy vs epochs ({config_name})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
