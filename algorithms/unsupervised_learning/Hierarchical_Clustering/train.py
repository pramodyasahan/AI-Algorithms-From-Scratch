import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from hierarchical_clustering import HierarchicalClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load Iris dataset
iris = load_iris()
X_iris = iris.data

# Loop over a range of n_clusters
for n_clusters in range(2, 6):
    # Train Hierarchical Clustering model for each n_clusters
    hierarchical_clustering = HierarchicalClustering(n_clusters=n_clusters, linkage='single')
    hierarchical_clustering.fit(X_iris)
    labels = hierarchical_clustering.labels_

    # Evaluate the clustering performance using Silhouette Score
    silhouette_avg = silhouette_score(X_iris, labels)

    print(f"n_clusters = {n_clusters}, Silhouette Score: {silhouette_avg:.2f}")

    # Visualize the clustering
    plt.figure(figsize=(8, 6))
    plt.scatter(X_iris[:, 0], X_iris[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50,
                label='Data Points')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(f'Hierarchical Clustering on Iris Dataset (n_clusters = {n_clusters})')
    plt.legend()
    plt.show()

# Plot the dendrogram only once, as it shows the entire clustering process
Z = linkage(X_iris, method='single')  # 'single' corresponds to the single linkage used earlier

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index or Cluster Size')
plt.ylabel('Distance')
plt.show()
