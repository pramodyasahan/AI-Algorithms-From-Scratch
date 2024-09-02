import numpy as np

# Setting a seed for reproducibility
np.random.seed(42)

# Network architecture parameters
INPUT_SIZE = 784  # Number of input features (e.g., 28x28 pixels for MNIST)
HIDDEN_SIZE = 10  # Number of neurons in the hidden layer
OUTPUT_SIZE = 10  # Number of classes (e.g., 10 digits for MNIST)

# Initialize weights and biases
W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.01  # Shape: (784, 10)
W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.01  # Shape: (10, 10)

b1 = np.zeros((1, HIDDEN_SIZE))  # Shape: (1, 10)
b2 = np.zeros((1, OUTPUT_SIZE))  # Shape: (1, 10)


def sigmoid(z):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Derivative of the sigmoid function.
    Used for backpropagation.
    """
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    """
    Softmax activation function for output layer.
    """
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_propagation(X):
    """
    Perform forward propagation to compute the activations.
    Parameters:
    - X: Input data, shape (number of examples, INPUT_SIZE)

    Returns:
    - Z1: Linear combination for hidden layer, shape (number of examples, HIDDEN_SIZE)
    - A1: Activation for hidden layer, shape (number of examples, HIDDEN_SIZE)
    - Z2: Linear combination for output layer, shape (number of examples, OUTPUT_SIZE)
    - A2: Activation for output layer, shape (number of examples, OUTPUT_SIZE)
    """
    # Linear combination for the hidden layer
    Z1 = np.dot(X, W1) + b1  # Shape: (m, 10)
    A1 = sigmoid(Z1)  # Activation for hidden layer, Shape: (m, 10)

    # Linear combination for the output layer
    Z2 = np.dot(A1, W2) + b2  # Shape: (m, 10)
    A2 = softmax(Z2)  # Activation for output layer, Shape: (m, 10)

    return Z1, A1, Z2, A2


def cross_entropy_loss(Y, A2):
    """
    Compute the cross-entropy loss.
    Parameters:
    - Y: True labels, shape (number of examples)
    - A2: Predicted probabilities from the output layer, shape (number of examples, OUTPUT_SIZE)

    Returns:
    - loss: Cross-entropy loss, a scalar.
    """
    m = Y.shape[0]  # Number of examples
    log_likelihood = -np.log(A2[range(m), Y])
    loss = np.sum(log_likelihood) / m
    return loss


def backpropagation(X, Y, Z1, A1, Z2, A2):
    """
    Perform backpropagation to compute gradients.
    Parameters:
    - X: Input data, shape (number of examples, INPUT_SIZE)
    - Y: True labels, shape (number of examples)
    - Z1: Linear combination for hidden layer
    - A1: Activation for hidden layer
    - Z2: Linear combination for output layer
    - A2: Activation for output layer

    Returns:
    - Gradients: dW1, db1, dW2, db2
    """
    m = X.shape[0]  # Number of examples

    # Output layer error
    dZ2 = A2
    dZ2[range(m), Y] -= 1
    dZ2 /= m

    # Gradients for W2 and b2
    dW2 = np.dot(A1.T, dZ2)  # Shape: (HIDDEN_SIZE, OUTPUT_SIZE)
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # Shape: (1, OUTPUT_SIZE)

    # Hidden layer error
    dA1 = np.dot(dZ2, W2.T)  # Shape: (m, HIDDEN_SIZE)
    dZ1 = dA1 * sigmoid_derivative(Z1)  # Shape: (m, HIDDEN_SIZE)

    # Gradients for W1 and b1
    dW1 = np.dot(X.T, dZ1)  # Shape: (INPUT_SIZE, HIDDEN_SIZE)
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # Shape: (1, HIDDEN_SIZE)

    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    """
    Update network parameters using gradient descent.
    """
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2


def train(X, Y, epochs, learning_rate):
    """
    Train the neural network using gradient descent.
    Parameters:
    - X: Input data, shape (number of examples, INPUT_SIZE)
    - Y: True labels, shape (number of examples)
    - epochs: Number of iterations for training
    - learning_rate: Learning rate for gradient descent
    """
    global W1, b1, W2, b2

    for epoch in range(epochs):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_propagation(X)
        # Compute loss
        loss = cross_entropy_loss(Y, A2)
        # Backpropagation
        dW1, db1, dW2, db2 = backpropagation(X, Y, Z1, A1, Z2, A2)
        # Update parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
