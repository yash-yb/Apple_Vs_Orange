import json
from typing import Tuple

import numpy as np
from PIL import Image

from CNN_gen1 import preprocess_image_array, IMAGE_SIZE
  

MODEL_PATH = "model_arch_data.json"


def load_weights(path: str = MODEL_PATH) -> dict:
    try:
        with open(path, "r") as model_values:
            data = json.load(model_values)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{path}' not found. Train the model first.")
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON from '{path}'.")

    # Match keys written by CNN_gen1.py: W1, B1, W2, B2
    W1 = np.array(data["W1"], dtype=np.float32)
    b1 = np.array(data["B1"], dtype=np.float32)
    W2 = np.array(data["W2"], dtype=np.float32)
    b2 = np.array(data["B2"], dtype=np.float32)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Use the exact same preprocessing as training / fine-tuning:
    RGB conversion, resize to IMAGE_SIZE, normalize to [0, 1], flatten.
    """
    arr_flat = preprocess_image_array(image_path, IMAGE_SIZE)
    return arr_flat.reshape(1, -1)


def predict_proba(image_path: str, params: dict) -> float:
    x = preprocess_image(image_path)

    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # probability of apple (label 1)

    return float(a2.squeeze())


def classify(image_path: str, threshold: float = 0.5) -> None:
    params = load_weights()
    prob_apple = predict_proba(image_path, params)
    prob_orange = 1.0 - prob_apple

    print(f"Apple probability:  {prob_apple:.4f}")
    print(f"Orange probability: {prob_orange:.4f}")

    if prob_apple >= threshold:
        print("Prediction: Apple")
    else:
        print("Prediction: Orange")


def main() -> None:
    image_path = input("Enter image path to classify: ").strip()
    if not image_path:
        print("No image path provided.")
        return

    try:
        classify(image_path)
    except Exception as exc:
        print(f"Error during classification: {exc}")


if __name__ == "__main__":
    main()