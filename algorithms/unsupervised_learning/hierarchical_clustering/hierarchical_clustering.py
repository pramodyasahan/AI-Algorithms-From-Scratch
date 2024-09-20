import numpy as np


class HierarchicalClustering:
    def __init__(self, n_clusters=3, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X):
        # Start with each point as its own cluster
        clusters = [[i] for i in range(len(X))]
        distances = self._compute_distances(X)

        while len(clusters) > self.n_clusters:
            # Find the closest clusters to merge
            i, j = self._find_closest_clusters(distances)
            clusters[i].extend(clusters[j])
            del clusters[j]

            # Update distances matrix
            distances = self._update_distances(distances, i, j, clusters)

        # Assign labels based on cluster membership
        self.labels_ = np.zeros(len(X), dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                self.labels_[index] = cluster_id

    def _compute_distances(self, X):
        n_samples = len(X)
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = np.linalg.norm(X[i] - X[j])
                distances[j, i] = distances[i, j]
        return distances

    def _find_closest_clusters(self, distances):
        min_dist = float('inf')
        closest_pair = (0, 0)
        for i in range(len(distances)):
            for j in range(i + 1, len(distances)):
                if distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    closest_pair = (i, j)
        return closest_pair

    def _update_distances(self, distances, i, j, clusters):
        for k in range(len(distances)):
            if k != i and k != j:
                if self.linkage == 'single':
                    distances[i, k] = min(distances[i, k], distances[j, k])
                    distances[k, i] = distances[i, k]
        distances = np.delete(distances, j, axis=0)
        distances = np.delete(distances, j, axis=1)
        return distances
