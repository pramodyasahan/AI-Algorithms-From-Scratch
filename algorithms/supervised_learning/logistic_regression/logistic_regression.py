import numpy as np


# Sigmoid activation function
def sigmoid(z):
    """
    Compute the sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100):
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate : float, optional (default=0.01)
            The learning rate for gradient descent.
        num_iterations : int, optional (default=100)
            The number of iterations for gradient descent.
        """
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Training input data.
        y : array-like of shape (n_samples,)
            Training target values (binary: 0 or 1).
        """
        # Get the number of samples and features
        n_samples, n_features = X.shape

        # Reshape y to ensure it is a column vector
        y = y.reshape(-1, 1)

        # Initialize weights randomly and bias to 0
        self.weights = np.random.randn(n_features, 1)
        self.bias = 0

        # Perform gradient descent for a specified number of iterations
        for _ in range(self.num_iterations):
            # Calculate the linear predictions: Xw + b
            linear_pred = np.dot(X, self.weights) + self.bias

            # Apply the sigmoid function to get probabilities
            y_pred = sigmoid(linear_pred)

            # Compute the gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict binary labels for input data X.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input data to predict.

        Returns:
        array-like of shape (n_samples,)
            Predicted binary labels (0 or 1).
        """
        # Compute the linear prediction
        linear_pred = np.dot(X, self.weights) + self.bias

        # Apply the sigmoid function to get predicted probabilities
        y_pred = sigmoid(linear_pred)

        # Convert probabilities to binary outputs (0 or 1)
        return (y_pred > 0.5).astype(int).flatten()
