import os
import gzip
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
X_test = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
y_test = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

results = []

for n in param_grid['n_estimators']:
    for md in param_grid['max_depth']:
        for mss in param_grid['min_samples_split']:
            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=md,
                min_samples_split=mss,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append([n, md, mss, acc])
            print("one result appended")

results_df = pd.DataFrame(results, columns=['n_estimators', 'max_depth', 'min_samples_split', 'accuracy'])
print(results_df)

best_row = results_df.loc[results_df['accuracy'].idxmax()]
print("\nBest parameters:")
print(best_row)