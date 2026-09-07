# LAB05 Image Classification with Support Vector Machine (SVM)

LAB05 covers supervised learning and computer vision using **Support Vector Machine (SVM)** with **Principal Component Analysis (PCA)** for image recognition and classification.

## Dataset
* **Dataset:** Bike vs. Car (DataImages)
* **Link:** https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset 

## Structure

```text
LAB05/
│
├── DataImages/
│   ├── Bike/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   │
│   └── Car/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
│
├── classification/
│   ├── main.py
│   ├── test_svm.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── svm_model.py
│   ├── evaluate.py
│   └── outputs/
│       ├── features.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── scaler.pkl
│       ├── svm_model.pkl
│       └── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── link-data.txt
```

## Topic
- **Part 1: Data Loading & Preprocessing**
    - **Image Reading & Cleaning:** Automatically detects class folders (`Cat`, `Dog`), resizes images to `100x100`, converts images to grayscale, and skips corrupted or unreadable images.
    - **Feature Extraction & Normalization:** Flattens 2D grayscale image arrays into 1D feature vectors ($100 \times 100 = 10,000$ features) and scales pixel values to $[0, 1]$.
    - **Dataset Splitting:**split training ($80\%$) and testing ($20\%$) sets.

- **Part 2: Dimensionality Reduction & Model Training**
    - **Pipeline Construction:** Combines `StandardScaler` and **Principal Component Analysis (PCA)** ($n\_components=150$) into a single Scikit-Learn `Pipeline`.
    - **Dimensionality Reduction:** Reduces feature dimensions from 10,000 to 150 components, preserving key variance while drastically reducing computational complexity for Support Vector Classification.
    - **SVM Model Training:** Trains a Support Vector Classifier (`SVC`) with an **RBF kernel** ($C=10, \gamma=\text{'scale'}$).
    - **Artifact Serialization:** Saves trained pipeline, model (`svm_model.pkl`), scaler (`scaler.pkl`), and cached preprocessed NumPy arrays into the `outputs/` folder.

- **Part 3: Evaluation & Visualization**
    - **Performance Metrics:** Evaluates predictions using Accuracy Score, Classification Report (Precision, Recall, F1-Score), and Confusion Matrix.
    - **Confusion Matrix Visualization:** Generates and exports heatmaps (`confusion_matrix.png`) to analyze class-wise misclassifications.
    - **Inference & Sampling:** Runs `test_svm.py` to pick random test samples, predict classes, and display side-by-side visual predictions (`prediction_sample.png`).

## Setup & Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the SVM Model:**
   ```bash
   python classification/main.py
   ```

3. **Test & Visualize Predictions:**
   ```bash
   python classification/test_svm.py
   ```

## Notes
- **Key Concepts & Takeaways:**
    - **Dimensionality Reduction (PCA):** Raw pixel matrices have high dimensionality ($10,000$ features per image). Applying PCA retains critical visual characteristics while significantly accelerating RBF kernel optimization.
    - **Pipeline Usage:** Encapsulating feature scaling and PCA transformation inside a single pipeline prevents data leakage from test sets into training steps.
    - **RBF Kernel Support Vector Machines:** Well-suited for non-linear decision boundaries in high-dimensional image feature spaces.