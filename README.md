# 👕 Fashion MNIST Neural Network — Built from Scratch with NumPy

A neural network that classifies clothing images from the **Fashion MNIST** dataset, implemented using **only NumPy**. No TensorFlow, Keras, or PyTorch: every forward pass, loss calculation, and gradient update is written by hand.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only-013243?logo=numpy&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-87.76%25-brightgreen)

---

## ✨ Highlights

- **Pure NumPy implementation**: all neural-network mathematics is done with matrix operations, no deep learning frameworks.
- **Hand-coded training loop**: forward propagation, loss computation, backpropagation, and parameter updates.
- **87.76% accuracy** on Fashion MNIST (10 clothing classes).
- Built to understand *how* neural networks work under the hood, not just how to call an API.

---

## 📊 Results

| Metric      | Value      |
|-------------|------------|
| Dataset     | Fashion MNIST (70,000 grayscale images, 28×28) |
| Classes     | 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) |
| Accuracy    | **87.76%** |

---

## 🧠 How It Works

1. **Data preparation**: images are flattened from 28×28 to 784 features and normalised to the 0–1 range.
2. **Forward pass**: inputs are multiplied by weight matrices, shifted by biases, and passed through activation functions.
3. **Loss**: predictions are compared with the true labels to measure error.
4. **Backpropagation**: gradients are computed with the chain rule using NumPy matrix operations.
5. **Parameter update**: weights and biases are adjusted with gradient descent over repeated epochs.

> 🔧 *Add your specifics here:* number of layers and neurons, activation functions, learning rate, epochs, and loss function.

---

## 🛠 Tech Stack

- **Python**
- **NumPy**: the only library used for the model's computations

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/saintcoder0/fashionmnist.git
cd fashionmnist

# Install the dependency
pip install numpy

# Run the notebook / script
jupyter notebook
```

---

## 📚 What I Learned

- How weights, biases, and activations combine to form a network
- The mathematics of backpropagation and gradient descent
- Vectorising computations with NumPy for speed and clarity
- Why frameworks exist, and what they abstract away

---

## 👤 Author

**Tushar Sharma**
[GitHub](https://github.com/saintcoder0) · [LinkedIn](https://www.linkedin.com/in/tushar-sharma-83b10a270/)
