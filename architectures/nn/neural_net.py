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
