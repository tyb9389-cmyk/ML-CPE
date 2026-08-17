import numpy as np

class KNNClusterAssigner:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, cluster_labels):
        self.X = np.array(X, dtype=np.float32)
        self.labels = np.array(cluster_labels, dtype=np.int32)
        self.n_clusters = int(cluster_labels.max()) + 1
        return self

    def predict(self, X_new):
        X_new = np.array(X_new, dtype=np.float32)
        diff = X_new[:, np.newaxis, :] - self.X[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))

        idx = np.argsort(dist, axis=1)[:, :self.k]
        neighbor_labels = self.labels[idx]

        preds = []
        for row in neighbor_labels:
            counts = np.bincount(row, minlength=self.n_clusters)
            preds.append(np.argmax(counts))

        return np.array(preds, dtype=np.int32)