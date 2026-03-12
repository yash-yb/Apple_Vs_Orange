# 🍎 Apple vs Orange Image Classifier

This project implements a **simple neural network from scratch using
NumPy** to classify images as **Apple or Orange**.

Unlike typical ML projects that use TensorFlow or PyTorch, this project
**builds the neural network manually**, including:

-   Forward propagation
-   Backpropagation
-   Gradient descent
-   Binary cross entropy loss

This makes it a good **educational project to understand how neural
networks work internally.**

------------------------------------------------------------------------

# 🚀 Features

-   Neural Network implemented **from scratch**
-   Uses **NumPy for calculations**
-   Uses **Pillow for image processing**
-   Binary classifier (**Apple vs Orange**)
-   Training from a dataset
-   **Single image fine‑tuning**
-   Model saving and loading
-   Simple CLI interface

------------------------------------------------------------------------

# 🧠 Model Architecture

Input Layer (3072 features)\
↓\
Hidden Layer (64 neurons, Sigmoid activation)\
↓\
Output Layer (1 neuron, Sigmoid)

Output:

-   **Closer to 1 → Apple**
-   **Closer to 0 → Orange**

Loss Function:

Binary Cross Entropy

Optimizer:

Gradient Descent

------------------------------------------------------------------------

# 📂 Dataset Structure

Your dataset should look like this:

project/ │ ├── ModelTuning/ │ └── Training/ │ ├── Apple/ │ │ ├──
apple1.jpg │ │ ├── apple2.png │ │ └── ... │ │ │ └── Orange/ │ ├──
orange1.jpg │ ├── orange2.png │ └── ... │ ├── model_arch_data.json ├──
main.py └── README.md

------------------------------------------------------------------------

# ⚙️ Installation

Install dependencies:

pip install numpy pillow

------------------------------------------------------------------------

# ▶️ Running the Program

Run:

python main.py

You will see:

Press A(start from scratch) or B(train on a single image)

------------------------------------------------------------------------

# 🏋️ Training From Scratch

Press:

A

The program will:

1.  Load all training images
2.  Train the neural network
3.  Print training progress
4.  Save the model to:

model_arch_data.json

Example output:

Epoch 0, BCE loss: 0.693\
Epoch 10, BCE loss: 0.612\
Training complete!

------------------------------------------------------------------------

# 🔧 Fine‑Tuning With One Image

Press:

B

Then enter:

Enter address of the file: test/apple1.jpg\
Enter Apple or Orange: Apple

The model will:

1.  Predict the image
2.  Compare with the correct label
3.  If wrong → update weights
4.  Save the improved model

Example:

Before update → prob_apple=0.42 predicted=Orange\
After update → prob_apple=0.61 predicted=Apple

------------------------------------------------------------------------

# 🖼 Image Processing

Each image is:

1.  Converted to **RGB**
2.  Resized to **32 × 32**
3.  Normalized to **0--1**
4.  Flattened to a vector

Final input size:

32 × 32 × 3 = **3072 features**

------------------------------------------------------------------------

# 📁 Model File

The trained model is stored in:

model_arch_data.json

Stored parameters:

-   W1
-   b1
-   W2
-   b2

------------------------------------------------------------------------

# ⚠️ Limitations

This is a **basic educational neural network**, so it does not include:

-   CNN layers
-   GPU acceleration
-   Batch training
-   Advanced optimizers

Accuracy depends heavily on dataset quality.

------------------------------------------------------------------------

# 💡 Possible Improvements

You could extend this project by adding:

-   Convolutional Neural Network (CNN)
-   Data augmentation
-   FastAPI prediction API
-   Web interface
-   Model evaluation metrics
-   Larger dataset

------------------------------------------------------------------------

# 📜 License

This project is open-source and free to use for learning purposes.
