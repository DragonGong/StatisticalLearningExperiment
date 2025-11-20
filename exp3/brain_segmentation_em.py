import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

img = np.loadtxt('exp3/brainimage.txt')
img_normalized = img / 255.0
pixels = img_normalized.flatten()
N = pixels.shape[0]
K = 3

kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(pixels.reshape(-1, 1))
mu = kmeans.cluster_centers_.flatten()
pi = np.ones(K) / K
sigma2 = np.var(pixels) * np.ones(K)

for iteration in range(30):
    gamma = np.zeros((N, K))
    for k in range(K):
        norm_pdf = np.exp(-0.5 * ((pixels - mu[k]) ** 2) / sigma2[k]) / np.sqrt(2 * np.pi * sigma2[k])
        gamma[:, k] = pi[k] * norm_pdf
    gamma_sum = gamma.sum(axis=1, keepdims=True)
    gamma /= gamma_sum

    pi = gamma.mean(axis=0)
    mu = (gamma.T @ pixels) / gamma.sum(axis=0)
    sigma2 = np.array([np.sum(gamma[:, k] * (pixels - mu[k])**2) / np.sum(gamma[:, k]) for k in range(K)])

labels = np.argmax(gamma, axis=1)
label_image = labels.reshape(img_normalized.shape)

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(img_normalized, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(label_image == 0, cmap='gray')
plt.title('Class 0: Outside Brain')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(label_image == 1, cmap='gray')
plt.title('Class 1: Gray Matter')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(label_image == 2, cmap='gray')
plt.title('Class 2: White Matter')
plt.axis('off')

plt.tight_layout()
plt.savefig("brain_segmentation_results.png", dpi=150, bbox_inches='tight')
print("\nSegmentation results saved as 'brain_segmentation_results.png'")