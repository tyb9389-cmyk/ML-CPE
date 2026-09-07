# LAB04 KNN Classification & K-Means Clustering
LAB04 covers supervised learning with **K-Nearest Neighbors (KNN)** for category classification and prediction

## Dataset
* **File:** `animal_dataset.csv`
* **Link:** `https://www.kaggle.com/datasets/shyamalb2/animal-dataset-csv`

## Structure

```~~text
LAB04/
│
├── classification/
│   ├── main.py
│   ├── data_loader.py
│   ├── knn_tf.py
│   ├── evaluate.py
│   └── outputs/
│       ├── 01_k_curve.png
│       ├── 02_confusion_matrix.png
│       └── predictions.csv
│
├── clustering/
│   ├── main.py
│   ├── data_loader.py
│   ├── kmeans_tf.py
│   ├── knn_tools.py
│   ├── visualize.py
│   └── outputs/
│       ├── 01_elbow.png
│       ├── 02_clusters.png
│       ├── cluster_summary.csv
│       └── clustered_animals.csv
│
├── data-animal/
│   └── animal_dataset.csv
│
├── link-data.txt
├── README.md
└── requirements.txt
```
## Topic
- Part 1: Classification (KNN)
    - Preparing Classification Data
        - Data Cleaning: checked no missing values
        - Label Encoding: target = Diet_Type (Carnivore, Herbivore, Omnivore)
        - Feature Scaling
        - Train/Val/Test Split: 60/20/20 ratio with stratify
    - Hyperparameter Tuning
        - Evaluated K values (1 to 15) on Validation Set
        - Plotted Validation Accuracy vs K curve (`01_k_curve.png`)
    - Model Training & Evaluation
        - Trained Custom KNN model using best K value
        - Plotted Confusion Matrix (`02_confusion_matrix.png`) and generated Classification Report
    - Baseline Comparison
        - Custom KNN vs Sklearn KNeighborsClassifier vs Majority Class Baseline
- Part 2: Clustering (K-Means)
    - Optimal K Selection (Elbow Method)
        - Evaluated K values (2 to 8) using Inertia and Silhouette Score
        - Selected N_CLUSTERS = 4 based on Elbow Curve (`01_elbow.png`)
    - Cluster Execution & Analysis
        - Ran K-Means until centroids stabilized
        - Plotted 2D Cluster Scatter Plot (`02_clusters.png`)
        - Exported average physical profiles to `cluster_summary.csv`
    - KNN Cluster Assignment
        - Used KNNClusterAssigner (K=5) on known clusters (800 rows) to classify new animals (200 rows)

## Notes
-   Key Concepts & Takeaways
    - KNN (Supervised): Requires labeled data to find the majority class among $K$ nearest neighbors
    - K-Means (Unsupervised): Groups unlabeled data by minimizing squared Euclidean distances to centroids