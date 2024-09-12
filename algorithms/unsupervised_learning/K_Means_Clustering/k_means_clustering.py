import numpy as np


class KMeansClustering:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # Initialize centroids randomly from the dataset
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Assign clusters based on the closest centroid
            clusters = self._assign_clusters(X)
            # Update centroids by calculating the mean of assigned points
            new_centroids = np.array([X[clusters == k].mean(axis=0) for k in range(self.n_clusters)])

            # If centroids do not change, we have converged
            if np.all(new_centroids == self.centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        # Assign clusters based on the closest centroid
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        return np.argmin(distances, axis=0)
