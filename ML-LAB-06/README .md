# LAB06 Neural Network (NN)

LAB06 covers apple leaf images classification for healthy and diseased, preprocessing, split dataset, train model with neural network (NN), prediction leaf healthy, and showing evaluation results.   

## Dataset
* **Dataset:** Healthy vs Diseased (DataImages) select only apple leaf
* **Link:** https://github.com/spMohanty/PlantVillage-Dataset/tree/master

## Structure

```text
LAB06/
│
├── DataImages/
│   ├── Diseased/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   │
│   └── Healthy/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
│
├── classification/
│   ├── main.py
│   ├── test_nn.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── nn_model.py
│   ├── evaluate.py
│   └── outputs/                   
│       ├── features.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       ├── y_test.npy
│       ├── nn_model.keras
│       ├── history.json
│       ├── confusion_matrix.png
│       ├── training_history.png
│       └── prediction_sample.png
│
├── requirements.txt
├── README.md
└── link-data.txt
```

## Topic
* **Apple Leaf Image Classification:** Binary classification to distinguish between Healthy and Diseased leaves.
* **Data Preprocessing:** Images are converted from BGR to RGB.
* **Neural Network model training:**
  * **Input Layer:** Accepts 100 x 100 x 3 RGB images, Scales pixel values from 0-255 down to 0-1 (Rescaling) for faster, stable convergence.
  * **Hidden Layers:** Fully connected layers (256, 128, 64 units) paired with ReLU activation, Batch Normalization, and Dropout (0.3–0.4) to mitigate overfitting.
  * **Output Layer:** Single unit with Sigmoid activation for binary classification.
* **Model Evaluation:** Performance measured using Accuracy, Classification Report, Confusion Matrix, and Accuracy/Loss history plots.

## Setup & Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the NN Model:**
   ```bash
   python classification/main.py
   ```

3. **Test & Visualize Predictions:**
   ```bash
   python classification/test_nn.py
   ```
