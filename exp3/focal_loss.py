import os
import gzip
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score

def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

data_dir = "exp2/assets/mnist"
X_train_full = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
y_train_full = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
X_test = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
y_test = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

# 人工制造不平衡：保留所有类，但只取 100 个 label=5 的样本
indices_to_keep = []
for label in range(10):
    idxs = np.where(y_train_full == label)[0]
    if label == 5:
        idxs = idxs[:100]  # 只取前 100 个 "5"
    indices_to_keep.append(idxs)

indices_to_keep = np.concatenate(indices_to_keep)
X_train_imb = X_train_full[indices_to_keep]
y_train_imb = y_train_full[indices_to_keep]

print("Imbalanced training set size:", len(y_train_imb))
unique, counts = np.unique(y_train_imb, return_counts=True)
print("Class counts:", dict(zip(unique, counts)))

# 实验 1: 不使用 class_weight（普通训练）
rf1 = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
rf1.fit(X_train_imb, y_train_imb)
y_pred1 = rf1.predict(X_test)

acc1 = accuracy_score(y_test, y_pred1)
bal_acc1 = balanced_accuracy_score(y_test, y_pred1)

# 实验 2: 使用 class_weight='balanced'
rf2 = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42
)
rf2.fit(X_train_imb, y_train_imb)
y_pred2 = rf2.predict(X_test)

acc2 = accuracy_score(y_test, y_pred2)
bal_acc2 = balanced_accuracy_score(y_test, y_pred2)

print("\n=== Without class_weight ===")
print(f"Accuracy: {acc1:.4f}")
print(f"Balanced Accuracy: {bal_acc1:.4f}")

print("\n=== With class_weight='balanced' ===")
print(f"Accuracy: {acc2:.4f}")
print(f"Balanced Accuracy: {bal_acc2:.4f}")

# 查看对少数类 "5" 的召回率（Recall）
report1 = classification_report(y_test, y_pred1, output_dict=True)
report2 = classification_report(y_test, y_pred2, output_dict=True)

recall_5_no_weight = report1['5']['recall']
recall_5_weighted = report2['5']['recall']

print(f"\nRecall for digit '5' (no weight): {recall_5_no_weight:.4f}")
print(f"Recall for digit '5' (with class_weight): {recall_5_weighted:.4f}")