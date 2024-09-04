import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        """
        Initialize the Linear Regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for training.
        """
        self.weights = None
        self.biases = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Fit the Linear Regression model to the training data.

        Parameters:
        X (ndarray): Training data features, shape (n_samples, n_features).
        y (ndarray): Training data labels, shape (n_samples, 1).
        """
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features, 1)
        self.biases = np.zeros(1)

        for _ in range(self.num_iterations):
            y_pred = np.dot(X, self.weights) + self.biases

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.biases -= self.learning_rate * db

    def predict(self, X):
        """
        Predict using the Linear Regression model.

        Parameters:
        X (ndarray): Input data features, shape (n_samples, n_features).

        Returns:
        ndarray: Predicted values, shape (n_samples, 1).
        """
        return np.dot(X, self.weights) + self.biases
