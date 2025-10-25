from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import time
from config import TRAIN_SAMPLES, RANDOM_STATE

def run_svm_experiments(X_train, y_train, X_test, y_test):
    print("=== SVM Parameter Tuning ===")
    X_train_sub = X_train[:TRAIN_SAMPLES]
    y_train_sub = y_train[:TRAIN_SAMPLES]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for C in [0.1, 1.0, 10.0]:
        clf = SVC(kernel='rbf', C=C, random_state=RANDOM_STATE, max_iter=10000)
        start = time.time()
        clf.fit(X_train_scaled, y_train_sub)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        t = time.time() - start
        key = f"RBF_C{C}"
        results[key] = {"accuracy": acc, "time": t}
        print(f"SVM (RBF, C={C}): Accuracy = {acc:.4f}, Time = {t:.2f}s")

    for kernel in ['linear', 'rbf', 'poly']:
        params = {'kernel': kernel, 'C': 1.0, 'random_state': RANDOM_STATE, 'max_iter': 10000}
        if kernel == 'poly':
            params['degree'] = 3
        clf = SVC(**params)
        start = time.time()
        clf.fit(X_train_scaled, y_train_sub)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        t = time.time() - start
        key = f"{kernel.upper()}_C1"
        results[key] = {"accuracy": acc, "time": t}
        print(f"SVM ({kernel}, C=1): Accuracy = {acc:.4f}, Time = {t:.2f}s")

    return results