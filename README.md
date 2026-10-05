# 🧠 ANN-MNIST-Digit-Classifier

A handwritten digit classifier built from scratch using **PyTorch** and deployed as an interactive **Streamlit web application**.

The user can draw a digit on a canvas, and the trained Artificial Neural Network predicts which digit (0–9) was drawn.

---

## 📌 Project Overview

This project was built to understand the complete workflow of a basic Deep Learning project:

- Loading and preprocessing image data
- Using PyTorch `Dataset` and `DataLoader`
- Building an Artificial Neural Network
- Training a neural network
- Evaluating model performance
- Saving and loading a trained PyTorch model
- Processing real-world user input
- Deploying the model using Streamlit

The model was trained on the **MNIST handwritten digit dataset** and achieved approximately **97% test accuracy**.

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Streamlit
- Streamlit Drawable Canvas
- Matplotlib

---

## 📂 Project Structure

```text
ANN-MNIST-Digit-Classifier/
│
├── notebooks/
│   └── mnist_ann_training.ipynb
│
├── src/
│   └── model.py
│
├── app.py
├── mnist_ann.pth
├── requirements.txt
└── README.md