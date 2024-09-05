import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from k_nearest_neighbours import KNN
from sklearn.datasets import make_blobs


def normalize(X, min_val, max_val):
    return (X - min_val) / (max_val - min_val)


X, y = make_blobs(n_samples=200, centers=2, cluster_std=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate the min and max for normalization from the training data
min_val = X_train.min(axis=0)
max_val = X_train.max(axis=0)

# Normalize the training and test data
X_train_normalized = normalize(X_train, min_val, max_val)
X_test_normalized = normalize(X_test, min_val, max_val)

# Create a DataFrame for plotting
df = pd.DataFrame(dict(x=X_train_normalized[:, 0], y=X_train_normalized[:, 1], label=y_train))
colors = {0: "green", 1: "orange"}

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
grouped = df.groupby('label')
for name, group in grouped:
    group.plot(ax=ax, kind="scatter", x="x", y="y", label=name, color=colors[name])
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.show()

# Initialize the KNN model
model = KNN(n_neighbors=3, distance="euclidian_distance")
model.fit(X_train_normalized, y_train)

# Make predictions on the normalized test data
y_pred = model.predict(X_test_normalized)

# Calculate and print the accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
