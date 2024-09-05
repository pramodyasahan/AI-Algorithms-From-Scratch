import numpy as np


def euclidian_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return np.sqrt(np.sum(np.square(point1 - point2)))


def manhattan_distance(point1, point2):
    # Calculate the Manhattan distance between two points
    return np.sum(np.abs(point1 - point2))


class KNN:
    def __init__(self, n_neighbors=3, distance="euclidian_distance"):
        # Initialize the KNN model with the number of neighbors
        self.y_train = None
        self.X_train = None
        self.distance = distance
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        # Store training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Predict labels for all test points
        return [self._predict(x) for x in X]

    def _predict(self, x):
        # Compute distances from x to all training points
        distances = ""
        if self.distance == "euclidian_distance":
            distances = [euclidian_distance(x_train, x) for x_train in self.X_train]
        elif self.distance == "manhattan_distance":
            distances = [manhattan_distance(x_train, x) for x_train in self.X_train]
        # Find the k closest points
        k_indices = np.argsort(distances)[:self.n_neighbors]
        # Get the labels of the k closest points
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return np.bincount(k_nearest_labels).argmax()
