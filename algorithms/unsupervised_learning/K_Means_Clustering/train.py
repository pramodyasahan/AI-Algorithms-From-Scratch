import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from k_means_clustering import KMeansClustering

# Load Iris dataset
iris = load_iris()
X_iris = iris.data

# Test different values of n_clusters to find the optimal number
inertia = []
silhouette_scores = []
cluster_range = range(2, 9)  # Trying from 2 to 8 clusters

for n_clusters in cluster_range:
    kmeans = KMeansClustering(n_clusters=n_clusters, max_iters=100)
    kmeans.fit(X_iris)
    labels = kmeans.predict(X_iris)

    # Calculate inertia (sum of squared distances to closest centroid)
    current_inertia = np.sum(
        [np.min(np.linalg.norm(X_iris - kmeans.centroids[label], axis=1) ** 2) for label in labels])
    inertia.append(current_inertia)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_iris, labels)
    silhouette_scores.append(silhouette_avg)

    # Visualize the clustering for the current number of clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_iris[:, 0], X_iris[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50,
                label='Data Points')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=200, linewidths=2,
                label='Centroids')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(f'KMeans Clustering on Iris Dataset with k={n_clusters}')
    plt.legend()
    plt.show()

# Plot Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o', label='Inertia')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', label='Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method for Optimal k')
plt.show()
