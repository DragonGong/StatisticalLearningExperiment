import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

data_dir = "exp2/assets/mnist"
X_train_img = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
y_train = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts, color='skyblue', edgecolor='black')
plt.title('Class Distribution (MNIST)')
plt.xlabel('Digit')
plt.ylabel('Number of Samples')
plt.xticks(unique)
for i, v in enumerate(counts):
    plt.text(i - 0.2, v + 200, str(v), fontsize=9)

plt.subplot(1, 2, 2)
pixel_values = X_train_img.flatten() / 255.0
plt.hist(pixel_values, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
plt.title('Pixel Intensity Distribution')
plt.xlabel('Normalized Pixel Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("mnist_eda_distribution.png", dpi=150, bbox_inches='tight')

np.random.seed(42)
plt.figure(figsize=(12, 5))
for i in range(10):
    idx = np.random.choice(np.where(y_train == i)[0])
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train_img[idx], cmap='gray')
    plt.title(f'Label: {i}')
    plt.axis('off')

plt.suptitle('Random Sample Images from Each Class', y=0.98)
plt.tight_layout()
plt.savefig("mnist_eda_samples.png", dpi=150, bbox_inches='tight')