import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

def plot_decision_tree(tree_model, max_depth=3, filename="tree.png"):
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model, max_depth=max_depth, filled=True, class_names=[str(i) for i in range(10)])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_kmeans_pca(X_pca, labels, filename="kmeans.png"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title("KMeans Clustering (PCA 2D)")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()