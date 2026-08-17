import numpy as np

class TFKMeans:
    def __init__(self, n_clusters=4, max_iter=100, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

    def _distance(self, X, centroids):
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))

    def fit(self, X):
        n_samples = X.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters ต้องไม่มากกว่าจำนวนข้อมูล")

        rng = np.random.default_rng(self.seed)
        start_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[start_idx].copy()

        for step in range(self.max_iter):
            dist = self._distance(X, centroids)
            labels = np.argmin(dist, axis=1)

            new_centroids = []
            for c in range(self.n_clusters):
                members = X[labels == c]
                if len(members) > 0:
                    new_centroids.append(members.mean(axis=0))
                else:
                    new_centroids.append(centroids[c])

            new_centroids = np.array(new_centroids)
            moved = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids

            if moved < 1e-4:
                break

        dist = self._distance(X, centroids)
        self.labels_ = np.argmin(dist, axis=1)
        self.centroids_ = centroids
        self.n_iter_ = step + 1

        min_dist = np.min(dist, axis=1)
        self.inertia_ = float(np.sum(min_dist**2))

        return self

    def fit_predict(self, X):
        return self.fit(X).labels_