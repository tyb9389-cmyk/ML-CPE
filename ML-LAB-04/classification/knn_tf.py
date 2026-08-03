import numpy as np


class TFKNNClassifier:

    def __init__(self, k=5):
        self.k = k

    # -----------------------------------------------------------------
    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y, dtype=np.int32)
        self.n_classes = int(y.max()) + 1
        return self

    # -----------------------------------------------------------------
    def _distance(self, X_new):
        """
        distance = sqrt( (x1-y1)² + (x2-y2)² + ... )
        """
        diff = X_new[:, None, :] - self.X_train[None, :, :]
        return np.sqrt(np.sum(np.square(diff), axis=2))

    # -----------------------------------------------------------------
    def predict(self, X):
        X = np.array(X, dtype=np.float32)

        # step 1 : distance
        dist = self._distance(X)

        # step 2 : select k nearest neighbors
        idx = np.argsort(dist, axis=1)[:, :self.k]
        neighbor_labels = self.y_train[idx]

        # step 3 : vote k
        predictions = []
        for row in neighbor_labels:
            counts = np.bincount(row, minlength=self.n_classes)
            predictions.append(np.argmax(counts))

        return np.array(predictions)

    # -----------------------------------------------------------------
    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))