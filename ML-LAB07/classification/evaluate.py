import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def evaluate_model(model, X_test, y_test, classes, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    preds = model.predict(X_test)
    if len(classes) == 2:
        y_pred = (preds > 0.5).astype("int32").flatten()
    else:
        y_pred = np.argmax(preds, axis=1)
        
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}")

def plot_training_history(history, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
   
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
  
    axs[0, 0].plot(epochs_range, acc, label='CNN Model (Train)', color='purple', linewidth=2)
    axs[0, 0].set_title('(a) Training accuracy of model', fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend(loc='lower right', fontsize=9)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    
  
    axs[0, 1].plot(epochs_range, val_acc, label='CNN Model (Val)', color='crimson', linewidth=2, marker='.')
    axs[0, 1].set_title('(b) Validation accuracy of model', fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Validation Accuracy')
    axs[0, 1].legend(loc='lower right', fontsize=9)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    

    axs[1, 0].plot(epochs_range, loss, label='CNN Model (Train Loss)', color='teal', linewidth=2)
    axs[1, 0].set_title('(c) Training Loss curve', fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend(loc='upper right', fontsize=9)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    

    axs[1, 1].plot(epochs_range, val_loss, label='CNN Model (Val Loss)', color='darkorange', linewidth=2, marker='.')
    axs[1, 1].set_title('(d) Validation Loss curve', fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend(loc='upper right', fontsize=9)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Research-style training history plot saved to {plot_path}")