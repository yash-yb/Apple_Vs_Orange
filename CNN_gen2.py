import os
import json
from typing import Tuple, List

from PIL import Image
import numpy as np

# Let apple be 1 and orange be 0.
# The model output is a probability between 0 and 1.

TRAIN_DATA_APPLE = "ModelTuning/Training/Apple"
TRAIN_DATA_ORANGE = "ModelTuning/Training/Orange"

APPLE_LABEL = 1
ORANGE_LABEL = 0

IMAGE_SIZE: Tuple[int, int] = (32, 32)
HIDDEN_SIZE1 = 64
HIDDEN_SIZE2 = 32
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01
EPOCHS = 100
MODEL_PATH = "model_arch_data.json"
EPS = 1e-7
L2_LAMBDA = 1e-4


def preprocess_image_array(
    file: str,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Load an image file of any common format (jpeg, png, etc.),
    convert to RGB, resize, normalize to [0, 1], and flatten.
    """
    img = Image.open(file).convert("RGB").resize(size)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    return img_array.flatten()


def load_image_folder(
    folder: str,
    label: int,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> Tuple[List[np.ndarray], List[int]]:
    """Load and flatten all images in a folder, returning data and labels lists."""
    data: List[np.ndarray] = []
    labels: List[int] = []

    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)

        # Skip directories and non-files
        if not os.path.isfile(file_path):
            continue

        try:
            img_flat = preprocess_image_array(file_path, size)
        except OSError:
            # Skip files that PIL cannot open
            continue

        data.append(img_flat)
        labels.append(label)

    return data, labels


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load apple and orange datasets and return X, y numpy arrays."""
    apple_data, apple_labels = load_image_folder(TRAIN_DATA_APPLE, APPLE_LABEL)
    orange_data, orange_labels = load_image_folder(TRAIN_DATA_ORANGE, ORANGE_LABEL)

    X = np.array(apple_data + orange_data, dtype=np.float32)
    y = np.array(apple_labels + orange_labels, dtype=np.float32).reshape(-1, 1)

    return X, y


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.
    Works correctly for large positive or negative values of x.
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given its output value."""
    return sigmoid_output * (1.0 - sigmoid_output)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(relu_output: np.ndarray) -> np.ndarray:
    return (relu_output > 0.0).astype(np.float32)


def initialize_parameters(input_size: int) -> dict:
    np.random.seed(42)

    # He initialization for ReLU layers
    W1 = np.random.randn(input_size, HIDDEN_SIZE1).astype(np.float32) * np.sqrt(
        2.0 / input_size
    )
    b1 = np.zeros((1, HIDDEN_SIZE1), dtype=np.float32)

    W2 = np.random.randn(HIDDEN_SIZE1, HIDDEN_SIZE2).astype(np.float32) * np.sqrt(
        2.0 / HIDDEN_SIZE1
    )
    b2 = np.zeros((1, HIDDEN_SIZE2), dtype=np.float32)

    # Output layer init can be smaller
    W3 = (np.random.randn(HIDDEN_SIZE2, OUTPUT_SIZE).astype(np.float32)) * 0.01
    b3 = np.zeros((1, OUTPUT_SIZE), dtype=np.float32)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def forward_pass(X: np.ndarray, params: dict) -> Tuple[dict, dict]:
    """Compute forward pass and return intermediate activations."""
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    W3, b3 = params["W3"], params["b3"]

    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)

    cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "a3": a3}
    return cache, {"y_hat": a3}


def compute_loss(y_hat: np.ndarray, y: np.ndarray, params: dict) -> float:
    """Binary cross-entropy loss + optional L2 weight decay."""
    y_hat_clipped = np.clip(y_hat, EPS, 1.0 - EPS)
    bce = -np.mean(
        y * np.log(y_hat_clipped) + (1.0 - y) * np.log(1.0 - y_hat_clipped)
    )
    m = y.shape[0]
    l2 = (
        L2_LAMBDA
        / (2.0 * m)
        * (
            float(np.sum(params["W1"] ** 2))
            + float(np.sum(params["W2"] ** 2))
            + float(np.sum(params["W3"] ** 2))
        )
    )
    return float(bce + l2)


def backward_pass(X: np.ndarray, y: np.ndarray, params: dict, cache: dict) -> dict:
    """Compute gradients of the parameters using backpropagation."""
    a1, a2, a3 = cache["a1"], cache["a2"], cache["a3"]
    W2, W3 = params["W2"], params["W3"]

    m = X.shape[0]

    # Output layer gradient
    dz3 = a3 - y
    dW3 = np.dot(a2.T, dz3) / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m
    dW3 += (L2_LAMBDA / m) * params["W3"]

    # Hidden layer 2 gradient (ReLU)
    dz2 = np.dot(dz3, W3.T) * relu_derivative(a2)
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    dW2 += (L2_LAMBDA / m) * params["W2"]

    # Hidden layer 1 gradient (ReLU)
    dz1 = np.dot(dz2, W2.T) * relu_derivative(a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    dW1 += (L2_LAMBDA / m) * params["W1"]

    return {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3,
    }


def update_parameters(params: dict, grads: dict, learning_rate: float = LEARNING_RATE) -> dict:
    """Gradient descent update."""
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    params["W3"] -= learning_rate * grads["dW3"]
    params["b3"] -= learning_rate * grads["db3"]
    return params


def save_model(params: dict, path: str = MODEL_PATH) -> None:
    model_arch_dict = {
        "W1": params["W1"].tolist(),
        "B1": params["b1"].tolist(),
        "W2": params["W2"].tolist(),
        "B2": params["b2"].tolist(),
        "W3": params["W3"].tolist(),
        "B3": params["b3"].tolist(),
    }
    with open(path, "w") as file:
        json.dump(model_arch_dict, file, indent=4)


def load_model(path: str = MODEL_PATH) -> dict:
    """Load model parameters from JSON and return them as numpy arrays."""
    with open(path, "r") as file:
        model_arch_dict = json.load(file)

    return {
        "W1": np.array(model_arch_dict["W1"], dtype=np.float32),
        "b1": np.array(model_arch_dict["B1"], dtype=np.float32),
        "W2": np.array(model_arch_dict["W2"], dtype=np.float32),
        "b2": np.array(model_arch_dict["B2"], dtype=np.float32),
        "W3": np.array(model_arch_dict["W3"], dtype=np.float32),
        "b3": np.array(model_arch_dict["B3"], dtype=np.float32),
    }


def load_single_image(file: str, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load a single image file and preprocess it like the training data."""
    img_flat = preprocess_image_array(file, size)
    return img_flat.reshape(1, -1)


def train() -> None:
    X, y = load_dataset()
    input_size = X.shape[1]

    params = initialize_parameters(input_size)

    for epoch in range(EPOCHS):
        cache, outputs = forward_pass(X, params)
        y_hat = outputs["y_hat"]

        loss = compute_loss(y_hat, y, params)
        mse = float(np.mean((y_hat - y) ** 2))

        grads = backward_pass(X, y, params, cache)
        params = update_parameters(params, grads)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, BCE loss: {loss:.6f}, MSE: {mse:.6f}")

    save_model(params)
    print("Training complete!")


def continue_training(
    file: str,
    label: int,
    learning_rate: float = 0.0005,
    steps: int = 5,
) -> None:
    """
    Fine-tune the existing model on a single labeled image.

    - file: path to the image
    - label: 1 for apple, 0 for orange
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. Train from scratch first."
        )

    params = load_model()
    X = load_single_image(file)
    y = np.array([[float(label)]], dtype=np.float32)

    # Show current prediction before update
    cache, outputs = forward_pass(X, params)
    y_hat_before = outputs["y_hat"]
    pred_before = int(y_hat_before[0, 0] >= 0.5)
    print(
        f"Before update -> prob_apple={y_hat_before[0,0]:.4f}, "
        f"predicted_label={pred_before}, true_label={label}"
    )

    # If the prediction is already correct, do not update the model
    if pred_before == label:
        print("Prediction is correct for this image. No fine-tuning applied.")
        return

    # Perform a small number of gradient steps on this single example
    for _ in range(steps):
        cache, outputs = forward_pass(X, params)
        grads = backward_pass(X, y, params, cache)
        params = update_parameters(params, grads, learning_rate=learning_rate)

    # Show prediction after update
    cache, outputs = forward_pass(X, params)
    y_hat_after = outputs["y_hat"]
    pred_after = int(y_hat_after[0, 0] >= 0.5)
    print(
        f"After update  -> prob_apple={y_hat_after[0,0]:.4f}, "
        f"predicted_label={pred_after}"
    )

    save_model(params)
    print("Single-image fine-tune complete and saved to JSON.")

def main() -> None:
    user_input = input(
        "Press A(start from scratch) or B(train on a single image) "
        "and anything else to exit: "
    ).strip().lower()
    if user_input == "a":
        train()
    elif user_input == "b":
        file = input("Enter address of the file: ").strip()
        label_str = input("Enter Apple or Orange: ").strip().lower()
        label = 1 if label_str == "apple" else 0  # 1 for apple and 0 for orange
        print(f"Label: {label}")
        continue_training(file, label)
    else:
        print("Training cancelled.")

if __name__ == "__main__":
    main()
