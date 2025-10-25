import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from config import TRAIN_SAMPLES, RANDOM_STATE, FIGURE_DIR
from utils.visualization import plot_kmeans_pca


def run_kmeans_experiments(X_train, y_train):
    print("=== KMeans: With vs Without PCA ===")
    X_sample = X_train[:TRAIN_SAMPLES]
    y_sample = y_train[:TRAIN_SAMPLES]

    results = {}

    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_sample)

    kmeans_raw = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
    labels_raw = kmeans_raw.fit_predict(X_sample)
    ari_raw = adjusted_rand_score(y_sample, labels_raw)
    results["KMeans_raw"] = {"ari": ari_raw}
    print(f"KMeans (raw 784D): ARI = {ari_raw:.4f}")
    plot_kmeans_pca(X_2d, labels_raw, os.path.join(FIGURE_DIR, "kmeans_raw_pca2d.png"))

    pca_95 = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca_95 = pca_95.fit_transform(X_sample)
    kmeans_pca95 = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
    labels_pca95 = kmeans_pca95.fit_predict(X_pca_95)
    ari_pca95 = adjusted_rand_score(y_sample, labels_pca95)
    results["KMeans_PCA95"] = {"ari": ari_pca95, "pca_dim": X_pca_95.shape[1]}
    print(f"KMeans (PCA 95% → {X_pca_95.shape[1]}D): ARI = {ari_pca95:.4f}")
    plot_kmeans_pca(X_2d, labels_pca95, os.path.join(FIGURE_DIR, "kmeans_pca95_pca2d.png"))

    pca_99 = PCA(n_components=0.99, random_state=RANDOM_STATE)
    X_pca_99 = pca_99.fit_transform(X_sample)
    kmeans_pca99 = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
    labels_pca99 = kmeans_pca99.fit_predict(X_pca_99)
    ari_pca99 = adjusted_rand_score(y_sample, labels_pca99)
    results["KMeans_PCA99"] = {"ari": ari_pca99, "pca_dim": X_pca_99.shape[1]}
    print(f"KMeans (PCA 99% → {X_pca_99.shape[1]}D): ARI = {ari_pca99:.4f}")
    plot_kmeans_pca(X_2d, labels_pca99, os.path.join(FIGURE_DIR, "kmeans_pca99_pca2d.png"))

    return results