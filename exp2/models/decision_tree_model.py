from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import time
import os
from config import TRAIN_SAMPLES, RANDOM_STATE, FIGURE_DIR
from utils.visualization import plot_decision_tree


def run_decision_tree_experiments(X_train, y_train, X_test, y_test):
    print("=== Decision Tree Parameter Tuning ===")
    X_train_sub = X_train[:TRAIN_SAMPLES]
    y_train_sub = y_train[:TRAIN_SAMPLES]

    results = {}
    for depth in [4, 8, 16]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        start = time.time()
        clf.fit(X_train_sub, y_train_sub)
        acc = accuracy_score(y_test, clf.predict(X_test))
        t = time.time() - start
        results[f"DT_depth{depth}"] = {"accuracy": acc, "time": t}
        print(f"Decision Tree (max_depth={depth}): Accuracy = {acc:.4f}, Time = {t:.2f}s")
        plot_decision_tree(clf, max_depth=16, filename=os.path.join(FIGURE_DIR, f"decision_tree_depth_{depth}.png"))

    return results