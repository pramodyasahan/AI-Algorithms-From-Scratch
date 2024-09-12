import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from k_nearest_neighbours import KNN
from sklearn.datasets import make_blobs


def normalize(X, min_val, max_val):
    return (X - min_val) / (max_val - min_val)


def plot_decision_boundary(X, y, model, ax, title):
    # Define the grid for plotting decision boundaries
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)  # Convert list to numpy array and reshape

    # Create a colormap
    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_points = ListedColormap(['#FF0000', '#0000FF'])

    # Plot decision boundaries
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap=cmap_points)
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_title(title)


# Generate synthetic data
X, y = make_blobs(n_samples=200, centers=2, cluster_std=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate the min and max for normalization from the training data
min_val = X_train.min(axis=0)
max_val = X_train.max(axis=0)

# Normalize the training and test data
X_train_normalized = normalize(X_train, min_val, max_val)
X_test_normalized = normalize(X_test, min_val, max_val)

# Create plots for each n_neighbors
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, n_neighbors in enumerate(range(2, 8)):
    # Initialize the KNN model with current n_neighbors
    model = KNN(n_neighbors=n_neighbors, distance="euclidian_distance")
    model.fit(X_train_normalized, y_train)

    # Make predictions on the normalized test data
    y_pred = model.predict(X_test_normalized)

    # Calculate and print the accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy with n_neighbors={n_neighbors}: {accuracy:.2f}")

    # Plot decision boundary
    plot_decision_boundary(X_train_normalized, y_train, model, axes[i], f'n_neighbors={n_neighbors}')

# Adjust layout
plt.tight_layout()
plt.show()
