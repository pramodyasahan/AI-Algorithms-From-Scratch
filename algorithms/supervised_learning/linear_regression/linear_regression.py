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
        y (ndarray): Training data labels, shape (n_samples,).

        Returns:
        self: Returns the instance of the model.
        """
        # Validate input dimensions
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")

        n_samples, n_features = X.shape

        # Reshape y to match output shape for consistency
        y = y.reshape(-1, 1)

        # Initialize parameters
        self.weights = np.random.randn(n_features, 1)
        self.biases = np.zeros((1,))

        for _ in range(self.num_iterations):
            # Make predictions
            y_pred = np.dot(X, self.weights) + self.biases

            # Compute gradients
            dw = (-2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (-2 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.biases -= self.learning_rate * db

        return self

    def predict(self, X):
        """
        Predict using the Linear Regression model.

        Parameters:
        X (ndarray): Input data features, shape (n_samples, n_features).

        Returns:
        ndarray: Predicted values, shape (n_samples,).
        """
        if self.weights is None or self.biases is None:
            raise ValueError("The model is not fitted yet. Call `fit` before `predict`.")

        return np.dot(X, self.weights) + self.biases.flatten()
