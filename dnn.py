"""Train a small neural network from scratch to classify MNIST digits.

Architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (sigmoid)
Dependency: numpy

Run: python digit_network.py
The four standard MNIST IDX files are downloaded into ./mnist_data/ on first run.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).with_name("mnist_data")
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def fetch_mnist_file(filename: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(BASE_URL + filename, path)
    return path


def load_images(filename: str) -> np.ndarray:
    with gzip.open(fetch_mnist_file(filename), "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file: {filename}")
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(count, rows * cols).astype(np.float32) / 255.0


def load_labels(filename: str) -> np.ndarray:
    with gzip.open(fetch_mnist_file(filename), "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file: {filename}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if len(labels) != count:
        raise ValueError(f"Label count mismatch in {filename}")
    return labels.astype(np.int64)


def one_hot(labels: np.ndarray, classes: int = 10) -> np.ndarray:
    result = np.zeros((len(labels), classes), dtype=np.float32)
    result[np.arange(len(labels)), labels] = 1.0
    return result


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Clipping keeps exp() in a numerically safe range.
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class DigitNetwork:
    """Fully connected digit network with a configurable layer structure."""

    def __init__(
        self,
        layer_sizes: tuple[int, ...] | list[int] = (784, 128, 64, 10),
        activations: tuple[str, ...] | list[str] = ("relu", "relu", "sigmoid"),
        seed: int = 1,
    ) -> None:
        if len(layer_sizes) < 2 or len(activations) != len(layer_sizes) - 1:
            raise ValueError("Provide one activation for each connection between layers.")
        if layer_sizes[0] != 784 or layer_sizes[-1] != 10:
            raise ValueError("The MNIST network must have 784 inputs and 10 outputs.")
        if any(size < 1 for size in layer_sizes):
            raise ValueError("Layer sizes must be positive.")
        if any(name not in ("relu", "sigmoid") for name in activations):
            raise ValueError("Activations must be 'relu' or 'sigmoid'.")

        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.activations = tuple(activations)
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for fan_in, fan_out, activation in zip(
            self.layer_sizes[:-1], self.layer_sizes[1:], self.activations
        ):
            scale = np.sqrt((2.0 if activation == "relu" else 1.0) / fan_in)
            self.weights.append(
                (rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32)
            )
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

    @staticmethod
    def _activate(z: np.ndarray, name: str) -> np.ndarray:
        return relu(z) if name == "relu" else sigmoid(z)

    @staticmethod
    def _activation_gradient(z: np.ndarray, a: np.ndarray, name: str) -> np.ndarray:
        return (z > 0.0).astype(np.float32) if name == "relu" else a * (1.0 - a)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[list[np.ndarray], list[np.ndarray]]]:
        activations = [x]
        zs = []
        a = x
        for weight, bias, activation in zip(self.weights, self.biases, self.activations):
            z = a @ weight + bias
            a = self._activate(z, activation)
            zs.append(z)
            activations.append(a)
        return a, (zs, activations)

    def train_batch(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        prediction, (zs, layer_activations) = self.forward(x)
        batch_size = len(x)

        # Mean squared error: L = 1/2 * sum((prediction - target)^2),
        # averaged over the batch for stable updates.
        loss = float(np.mean(np.sum(0.5 * (prediction - y) ** 2, axis=1)))

        delta = (prediction - y) * self._activation_gradient(
            zs[-1], prediction, self.activations[-1]
        )
        for layer in range(len(self.weights) - 1, -1, -1):
            dw = layer_activations[layer].T @ delta / batch_size
            db = np.mean(delta, axis=0, keepdims=True)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * self._activation_gradient(
                    zs[layer - 1], layer_activations[layer], self.activations[layer - 1]
                )
            self.weights[layer] -= learning_rate * dw
            self.biases[layer] -= learning_rate * db
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores, _ = self.forward(x)
        return np.argmax(scores, axis=1)

    def save(self, path: str | Path) -> None:
        """Save learned weights and biases so another program can reuse them."""
        values = {f"w{i}": weight for i, weight in enumerate(self.weights)}
        values.update({f"b{i}": bias for i, bias in enumerate(self.biases)})
        values["layer_sizes"] = np.asarray(self.layer_sizes, dtype=np.int64)
        values["activations"] = np.asarray(self.activations, dtype="U8")
        np.savez(path, **values)

    def load(self, path: str | Path) -> None:
        """Load parameters and restore the saved architecture."""
        with np.load(path) as data:
            self.layer_sizes = tuple(int(size) for size in data["layer_sizes"])
            self.activations = tuple(str(name) for name in data["activations"])
            self.weights = [data[f"w{i}"].copy() for i in range(len(self.activations))]
            self.biases = [data[f"b{i}"].copy() for i in range(len(self.activations))]


def accuracy(model: DigitNetwork, images: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(model.predict(images) == labels))


def main() -> None:
    train_x = load_images(FILES["train_images"])
    train_labels = load_labels(FILES["train_labels"])
    test_x = load_images(FILES["test_images"])
    test_labels = load_labels(FILES["test_labels"])
    train_y = one_hot(train_labels)

    model = DigitNetwork()
    epochs = 8
    batch_size = 64
    learning_rate = 0.05
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        order = rng.permutation(len(train_x))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            losses.append(model.train_batch(train_x[indices], train_y[indices], learning_rate))

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss {np.mean(losses):.4f} | "
            f"test accuracy {accuracy(model, test_x, test_labels):.2%}"
        )

    model_path = Path(__file__).with_name("digit_model.npz")
    model.save(model_path)
    print(f"Saved trained model to {model_path}")

    # Show a few example predictions from the test set.
    predictions = model.predict(test_x[:10])
    print("\nFirst 10 test labels:     ", test_labels[:10].tolist())
    print("First 10 predictions:     ", predictions.tolist())


if __name__ == "__main__":
    from dnn_gui import launch

    launch()
