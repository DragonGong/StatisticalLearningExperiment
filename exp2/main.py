from data.loader import load_mnist
from models.svm_model import run_svm_experiments
from models.decision_tree_model import run_decision_tree_experiments
from models.kmeans_clustering import run_kmeans_experiments

def main():
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    svm_results = run_svm_experiments(X_train, y_train, X_test, y_test)
    dt_results = run_decision_tree_experiments(X_train, y_train, X_test, y_test)
    kmeans_results = run_kmeans_experiments(X_train, y_train)

    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    for name, res in svm_results.items():
        print(f"SVM {name}: Acc={res['accuracy']:.4f}, Time={res['time']:.2f}s")
    for name, res in dt_results.items():
        print(f"DT {name}: Acc={res['accuracy']:.4f}, Time={res['time']:.2f}s")
    for name, res in kmeans_results.items():
        if 'pca_dim' in res:
            print(f"{name}: ARI={res['ari']:.4f} (dim={res['pca_dim']})")
        else:
            print(f"{name}: ARI={res['ari']:.4f}")

if __name__ == "__main__":
    main()