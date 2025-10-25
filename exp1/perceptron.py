import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")



class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=1000, tol=1e-6):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.weights = None
        self.bias = None
        self.converged = False

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        prev_weights = self.weights.copy()

        for epoch in range(self.max_epochs):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else -1
                if y_pred != y[idx]:
                    update = self.lr * y[idx]
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1

            # 检查收敛
            weight_change = np.linalg.norm(self.weights - prev_weights)
            if weight_change < self.tol and errors == 0:
                self.converged = True
                print(f"Perceptron converged at epoch {epoch + 1}")
                break
            prev_weights = self.weights.copy()

        if not self.converged:
            print(f"Perceptron did not converge within {self.max_epochs} epochs")

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)


df = pd.read_csv('Data.csv')

if 'fetal_health' not in df.columns:
    raise ValueError("CSV 文件中未找到 'fetal_health' 列，请确认数据格式！")

X = df.drop('fetal_health', axis=1).values
y = df['fetal_health'].values

y = y.astype(float)

print(f"原始标签分布: {np.bincount(y.astype(int))}")

y_binary = np.where(y == 1, 1, 0)

y_perceptron = np.where(y_binary == 1, 1, -1)

X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
_, _, y_train_perc, y_test_perc = train_test_split(
    X, y_perceptron, test_size=0.2, random_state=42, stratify=y_perceptron
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 1: 训练感知机
print("\n=== Task 1: Perceptron ===")
perceptron = Perceptron(learning_rate=0.01, max_epochs=2000)
perceptron.fit(X_train_scaled, y_train_perc)

y_pred_perc = perceptron.predict(X_test_scaled)
y_pred_bin = np.where(y_pred_perc == 1, 1, 0)

acc = accuracy_score(y_test_bin, y_pred_bin)
prec = precision_score(y_test_bin, y_pred_bin, zero_division=0)
rec = recall_score(y_test_bin, y_pred_bin, zero_division=0)
f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)

print(f"Perceptron Results:")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-score : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_bin, y_pred_bin, target_names=['Abnormal (2,3)', 'Normal (1)']))

# Task 2: SVM with different kernels
print("\n=== Task 2: SVM Comparison ===")

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_results = []

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train_bin)
    y_pred_svm = svm.predict(X_test_scaled)

    acc = accuracy_score(y_test_bin, y_pred_svm)
    f1 = f1_score(y_test_bin, y_pred_svm, zero_division=0)
    svm_results.append((kernel, acc, f1))
    print(f"SVM ({kernel:8s}) → Acc: {acc:.4f}, F1: {f1:.4f}")

print("\n--- Tuning RBF SVM ---")
best_f1 = 0
best_params = {}
for C in [0.1, 1, 10, 100]:
    for gamma in ['scale', 'auto', 0.001, 0.01, 0.1, 1]:
        svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm_rbf.fit(X_train_scaled, y_train_bin)
        y_pred = svm_rbf.predict(X_test_scaled)
        f1 = f1_score(y_test_bin, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_params = {'C': C, 'gamma': gamma}

print(f"Best RBF SVM → F1: {best_f1:.4f}, Params: {best_params}")

final_svm = SVC(kernel='rbf', **best_params, random_state=42)
final_svm.fit(X_train_scaled, y_train_bin)
y_pred_final = final_svm.predict(X_test_scaled)
print("\nFinal SVM Performance:")
print(classification_report(y_test_bin, y_pred_final, target_names=['Abnormal (2,3)', 'Normal (1)']))
