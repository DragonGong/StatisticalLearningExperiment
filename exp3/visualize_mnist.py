# first step
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import umap # note: macos arm chip, dev is on ubuntu vm machine , error: Illegal instruction (core dumped) , unsolved yet.
def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

data_dir = "exp2/assets/mnist"
X_train = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
y_train = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))

n_samples = 5000
X_subset = X_train[:n_samples]
y_subset = y_train[:n_samples]
print(f"Using {n_samples} samples for visualization.")

print("Running PCA...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_subset)

print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca')
X_tsne = tsne.fit_transform(X_subset)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for i in range(10):
    mask = y_subset == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], label=str(i), s=8)
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[i]], label=str(i), s=8)

axes[0].set_title('PCA')
axes[1].set_title('t-SNE')
for ax in axes:
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, linestyle='--', alpha=0.5)

axes[1].legend(title="Digit", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("mnist_pca_tsne.png", dpi=150, bbox_inches='tight')
print("Figure saved as mnist_pca_tsne.png")
plt.show()