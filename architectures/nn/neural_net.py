import numpy as np
from torchvision.datasets import MNIST

train_data = MNIST("data", train=True, download=True)
test_data = MNIST("data", train=False, download=True)

X_train = np.array(train_data.data.reshape(60000, 784))
print(f"Size of X_train: {X_train.shape}")

y_train = np.array(train_data.targets)
print(f"Size of y_train: {y_train.shape}")

INPUT_SIZE = 784
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10

W1 = np.random.randn(HIDDEN_SIZE, INPUT_SIZE) * 0.01
b1 = np.zeros((1, HIDDEN_SIZE))
W2 = np.random.randn(OUTPUT_SIZE, HIDDEN_SIZE) * 0.01
b2 = np.zeros((1, OUTPUT_SIZE))

print(W1.shape, "\n", b1.shape, "\n", W2.shape, "\n", b2.shape)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2.T) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def relu_derivative(Z):
    return Z > 0


def backward_propagation(X, y, Z1, A1, Z2, W2, b2):
    m = y.shape[0]

    dZ2 = A2 - np.eye(10)[y]
    dW2 = np.dot(dZ2.T, A1) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2)
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = np.dot(dZ1.T, X) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)
backward_propagation(X_train, y_train, Z1, A1, Z2, W2, b2)
