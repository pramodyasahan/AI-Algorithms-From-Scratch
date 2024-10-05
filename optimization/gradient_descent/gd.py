import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(123)
X = np.random.rand(100, 2)
print(X.shape)

weights_true = np.array([5, 8])
bias_true = 4
y = np.dot(X, weights_true) + bias_true + np.random.randn(100)
print(y.shape)

plt.scatter(X[:, 1], y, color="blue", label="Feature1 vs Target")
plt.title("Feature1 vs Target")
plt.xlabel("Feature1")
plt.ylabel("Target")
plt.legend()
plt.show()


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def predict(X, weights, bias):
    return np.dot(X, weights) + bias


def compute_gradients(X, y_true, y_pred):
    num_samples = len(y_true)
    dw = -(2 / num_samples) * np.dot(X.T, (y_true - y_pred))
    db = -(2 / num_samples) * np.sum(y_true - y_pred)

    return dw, db


def update_parameters(weights, bias, dw, db, learning_rate):
    weights -= learning_rate * dw
    bias -= learning_rate * db

    return weights, bias


def gradient_descent(X, y_true, weights, bias, learning_rate, epochs):
    loss = []
    for epoch in range(epochs):
        y_pred = predict(X, weights, bias)
        dw, db = compute_gradients(X, y_true, y_pred)
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)
        loss.append(mse_loss(y_true, predict(X, weights, bias)))

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss[epoch]}")

    return weights, bias, loss


def plot_loss_curve(loss_list):
    plt.plot(range(len(loss_list)), loss_list, color="green")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.show()


def plot_3d_regression(X, y_true, weights, bias):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y_true, color="blue", label="Data Points")

    x0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    Y_pred = weights[0] * X0 + weights[1] * X1 + bias

    ax.plot_surface(X0, X1, Y_pred, color="red", alpha=0.5, label="Regression Plane")
    ax.set_title("3D Linear Regression")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Target")

    plt.legend()
    plt.show()


learning_rate = 0.01
epochs = 2000

weights = np.random.rand(2)
bias = np.random.rand()

final_weights, final_bias, loss_list = gradient_descent(
    X, y, weights, bias, learning_rate, epochs
)
print(
    f"Final Weights: {final_weights}, Final Bias: {final_bias}, Final Loss: {loss_list[-1]}"
)

plot_loss_curve(loss_list)
plot_3d_regression(X, y, final_weights, final_bias)
