import os
import gzip
from urllib.request import urlretrieve
import numpy as np

from config import DATA_DIR

def parse_images(data):
    magic = int.from_bytes(data[0:4], 'big')
    num_images = int.from_bytes(data[4:8], 'big')
    rows = int.from_bytes(data[8:12], 'big')
    cols = int.from_bytes(data[12:16], 'big')
    parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
    return parsed.reshape(num_images, rows * cols).astype(np.float32)

def parse_labels(data):
    magic = int.from_bytes(data[0:4], 'big')
    num_items = int.from_bytes(data[4:8], 'big')
    parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
    return parsed

def download_and_extract(filename):
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        urlretrieve(base_url + filename, path)
    with gzip.open(path, 'rb') as f:
        return f.read()

def load_mnist():
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    X_train = parse_images(download_and_extract(files["train_images"]))
    y_train = parse_labels(download_and_extract(files["train_labels"]))
    X_test = parse_images(download_and_extract(files["test_images"]))
    y_test = parse_labels(download_and_extract(files["test_labels"]))
    return X_train, y_train, X_test, y_test