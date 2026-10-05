# 🧠 ANN-MNIST-Digit-Classifier

A simple Artificial Neural Network (ANN) built using **PyTorch** to classify handwritten digits from the **MNIST dataset**.

This project was built as a hands-on deep learning practice project to understand the complete workflow of a neural network — from loading and preprocessing data to training, evaluation, saving the model, and deploying it with Streamlit.

---

## 📌 Project Overview

The goal of this project is to build an ANN that can recognize handwritten digits from **0 to 9**.

The trained model takes a `28 × 28` grayscale image as input and predicts which digit it represents.

The project also includes a **Streamlit web application** where users can draw a digit on a canvas and get a real-time prediction from the trained ANN.

### Project Pipeline

```text
MNIST Dataset
      ↓
DataLoader
      ↓
ANN Model
      ↓
Training
      ↓
Evaluation
      ↓
Save Model (.pth)
      ↓
Streamlit App
      ↓
Draw Digit
      ↓
Image Preprocessing
      ↓
ANN Prediction
      ↓
Predicted Digit
```

---

## 📊 Dataset

This project uses the **MNIST handwritten digit dataset**.

MNIST contains grayscale images of handwritten digits from `0` to `9`.

### Dataset Details

| Property | Value |
|---|---|
| Training images | 60,000 |
| Testing images | 10,000 |
| Image size | 28 × 28 |
| Channels | 1 |
| Classes | 10 |
| Labels | 0–9 |
| Image type | Grayscale |

Each image contains:

```text
28 × 28 = 784 pixels
```

Therefore, after flattening the image, the ANN receives **784 input features**.

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Deep Learning

- PyTorch
- Torchvision

### Data Processing

- NumPy
- Pillow

### Visualization

- Matplotlib

### Web Application

- Streamlit
- Streamlit Drawable Canvas

---

## 📂 Project Structure

```text
ANN-MNIST-Digit-Classifier/
│
├── data/
│   └── MNIST dataset
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
├── README.md
└── .gitignore
```

### File Description

| File | Purpose |
|---|---|
| `notebooks/mnist_ann_training.ipynb` | Model training and experimentation |
| `src/model.py` | ANN architecture |
| `app.py` | Streamlit application |
| `mnist_ann.pth` | Trained model weights |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `data/` | MNIST dataset |

---

# 🧠 ANN Architecture

The neural network used in this project has the following architecture:

```text
Input
784 neurons
   ↓
Linear Layer
784 → 128
   ↓
ReLU
   ↓
Linear Layer
128 → 64
   ↓
ReLU
   ↓
Linear Layer
64 → 10
   ↓
Output
10 classes
```

### Architecture Code

```python
import torch.nn as nn


class ANN(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(num_features, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
```

---

## 🔍 Why 784 Input Features?

MNIST images have dimensions:

```text
28 × 28
```

Therefore:

```text
28 × 28 = 784
```

The `Flatten()` layer converts:

```text
[1, 28, 28]
```

into:

```text
[784]
```

This allows the image to be passed into the first linear layer:

```text
784 → 128
```

---

# 📦 Loading the Dataset

Torchvision provides the MNIST dataset directly, so there was no need to manually create `X` and `y`.

```python
import torch
from torchvision import datasets
from torchvision.transforms import transforms


transform = transforms.ToTensor()


train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
```

---

# 🚚 DataLoader

The dataset is loaded using PyTorch's `DataLoader`.

```python
from torch.utils.data import DataLoader


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)


test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)
```

### Why DataLoader?

Instead of sending the entire dataset to the model at once, the data is divided into batches.

For example:

```text
60,000 images
      ↓
Batch 1 → 64 images
Batch 2 → 64 images
Batch 3 → 64 images
...
```

This makes training more efficient and manageable.

### Why `shuffle=True`?

The training data is shuffled after every epoch so that the model does not always see the samples in the same order.

For testing, shuffling is not necessary:

```python
shuffle=False
```

---

# ⚙️ Model Training

The model was trained using:

- **Loss Function:** Cross Entropy Loss
- **Optimizer:** SGD
- **Batch Size:** 64
- **Epochs:** 100

### Training Setup

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

```

### Training Loop

```python
for epoch in range(epochs):

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {loss.item()}"
    )
```

---

# 🔄 How Training Works

For every batch, the following steps happen.

### 1. Forward Pass

The images are passed through the ANN.

```text
Image
 ↓
Flatten
 ↓
784 → 128
 ↓
ReLU
 ↓
128 → 64
 ↓
ReLU
 ↓
64 → 10
```

The model produces 10 output values.

Each value corresponds to one digit:

```text
0 → score
1 → score
2 → score
...
9 → score
```

---

### 2. Calculate Loss

The predicted output is compared with the actual label.

```python
loss = criterion(outputs, labels)
```

Cross Entropy Loss is used because this is a multi-class classification problem.

---

### 3. Backpropagation

```python
loss.backward()
```

This calculates the gradients of the model parameters.

---

### 4. Update Weights

```python
optimizer.step()
```

The optimizer updates the model's weights to reduce the loss.

---

# 📈 Model Performance

After training, the model achieved approximately:

```text
Test Accuracy ≈ 97%
```

This shows that the ANN successfully learned to recognize handwritten digits from the MNIST dataset.

---

# 💾 Saving the Model

After training, the model weights were saved using PyTorch's `state_dict`.

```python
torch.save(
    model.state_dict(),
    "mnist_ann.pth"
)
```

The `.pth` file contains the trained parameters of the neural network.

---

# 🔄 Loading the Model

The saved model is loaded in the Streamlit application.

```python
model = ANN(784)

model.load_state_dict(
    torch.load(
        "mnist_ann.pth",
        map_location="cpu"
    )
)

model.eval()
```

`model.eval()` switches the model to evaluation mode.

---

# 🌐 Streamlit Application

The project includes a Streamlit application that allows users to draw a digit.

The application contains:

- Drawing canvas
- Predict button
- Clear button
- Processed MNIST input preview
- Digit prediction

### Application Flow

```text
User draws digit
       ↓
300 × 300 canvas
       ↓
Image preprocessing
       ↓
28 × 28 MNIST image
       ↓
PyTorch tensor
       ↓
ANN
       ↓
Prediction
```

---

# ✏️ Drawing Canvas

The application uses `streamlit-drawable-canvas`.

```python
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=300,
    height=300,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas",
)
```

The user draws a digit on a `300 × 300` canvas.

---

# 🖼️ Image Preprocessing

One of the most important parts of this project was preprocessing the user's drawing.

The MNIST model expects:

```text
28 × 28 grayscale image
```

But the Streamlit canvas produces:

```text
300 × 300 RGBA image
```

Therefore, the drawing cannot simply be resized directly to `28 × 28`.

The preprocessing pipeline became:

```text
300 × 300 Canvas
       ↓
RGBA → Grayscale
       ↓
Find Digit
       ↓
Crop Digit
       ↓
Resize
       ↓
Center Digit
       ↓
28 × 28 Image
       ↓
Tensor
       ↓
ANN
```

---

# 🧹 Digit Cropping

First, the non-black pixels are detected.

```python
coords = np.argwhere(
    image_array > 20
)
```

This helps find where the digit is located.

A bounding box is then created:

```python
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)
```

The digit is cropped:

```python
cropped = image.crop(
    (x_min, y_min, x_max + 1, y_max + 1)
)
```

---

# 📐 Resizing

The cropped digit is resized while maintaining its aspect ratio.

```python
cropped.thumbnail((20, 20))
```

The digit is kept smaller than the full `28 × 28` image to leave space around it.

---

# 🎯 Centering the Digit

A new black `28 × 28` image is created.

```python
processed_image = Image.new(
    "L",
    (28, 28),
    0
)
```

The digit is then placed in the center.

```python
x_offset = (28 - cropped.width) // 2
y_offset = (28 - cropped.height) // 2
```

This makes the input more similar to the images used during MNIST training.

---

# 🔢 Converting to Tensor

The processed image is converted into a PyTorch tensor.

```python
image_tensor = transforms.ToTensor()(
    processed_image
)
```

A batch dimension is then added:

```python
image_tensor = image_tensor.unsqueeze(0)
```

The final shape becomes:

```text
[1, 1, 28, 28]
```

This is the format expected by the ANN.

---

# 🎯 Prediction

The model produces output scores for all 10 digits.

```python
with torch.no_grad():

    output = model(image_tensor)
```

The predicted digit is selected using `argmax`.

```python
prediction = torch.argmax(
    output,
    dim=1
).item()
```

Finally:

```python
st.success(
    f"🎯 Prediction: {prediction}"
)
```

---

# 🚨 Problems Faced & Solutions

This project was not just about getting a model to work. Several problems were encountered while building the complete application.

---

## 1. Understanding Dataset vs X and y

### Problem

Initially, there was confusion about whether the MNIST dataset needed to be separated manually into:

```text
X → Images
y → Labels
```

### Solution

Torchvision's MNIST dataset already returns:

```python
image, label
```

When using:

```python
for images, labels in train_loader:
```

`images` acts as the input data and `labels` act as the target values.

Therefore, a custom dataset was not necessary.

---

# 2. Understanding Batches

### Problem

It was initially unclear why the DataLoader returns batches instead of individual images.

### Solution

The DataLoader divides the dataset into batches.

With:

```python
batch_size=64
```

each iteration processes up to 64 images.

```text
Dataset
 ↓
Batch of 64
 ↓
Model
 ↓
Loss
 ↓
Backpropagation
 ↓
Next batch
```

---

# 3. Understanding the 784 Input Features

### Problem

It was unclear why the first layer was:

```python
nn.Linear(784, 128)
```

### Solution

MNIST images are:

```text
28 × 28
```

Therefore:

```text
28 × 28 = 784
```

The `Flatten()` layer converts the image into 784 values.

---

# 4. Canvas Returned RGBA Instead of Grayscale

### Problem

The Streamlit canvas returned an image with shape:

```text
(300, 300, 4)
```

The four channels were:

```text
Red
Green
Blue
Alpha
```

### Solution

The image was converted to grayscale using Pillow:

```python
image = Image.fromarray(
    image
).convert("L")
```

Now the image contains a single grayscale channel.

---

# 5. Model Predicted Drawings Incorrectly

### Problem

The trained model achieved around 97% test accuracy, but predictions from the Streamlit drawing canvas were initially poor.

The first approach was:

```text
300 × 300 drawing
       ↓
Resize directly to 28 × 28
       ↓
Model
```

Even though the image visually looked correct, predictions were inconsistent.

### Why?

The model was trained on MNIST images that have a particular:

- Scale
- Position
- Centering
- Digit size

The user's drawing did not necessarily have the same distribution.

Therefore, the model was receiving inputs that were different from the data it had seen during training.

### Solution

A better preprocessing pipeline was implemented:

```text
Canvas
 ↓
Grayscale
 ↓
Find non-black pixels
 ↓
Bounding box
 ↓
Crop
 ↓
Resize while preserving aspect ratio
 ↓
Center in 28 × 28 image
 ↓
Tensor
 ↓
Prediction
```

This significantly improved the predictions.

---

# 🧠 Key Lessons Learned

Through this project, I learned the complete basic workflow of a deep learning project.

### PyTorch

- Creating neural networks with `nn.Module`
- Using `nn.Sequential`
- Linear layers
- ReLU activation
- Flattening images
- Loss functions
- Optimizers
- Backpropagation
- Training loops
- Evaluation
- Saving and loading models

### Dataset Handling

- Using torchvision datasets
- DataLoader
- Batch processing
- Training vs testing datasets
- Shuffling training data

### Computer Vision

- Image dimensions
- Grayscale conversion
- Image tensors
- Bounding boxes
- Cropping
- Resizing
- Centering images
- Image preprocessing

### Streamlit

- Creating interactive web applications
- Streamlit buttons
- Columns
- Image display
- Model caching
- Drawing canvas
- Connecting a trained ML model to a UI

---

# 🔑 Important Concept Learned

One of the biggest lessons from this project was:

> **A good model can still give bad predictions if the input preprocessing does not match the training data.**

The model had approximately 97% accuracy on MNIST, but initially performed poorly on hand-drawn inputs.

The problem was not necessarily the model.

The problem was the difference between:

```text
Training Data
      vs
Real-world Input
```

Making the preprocessing pipeline more similar to the training data greatly improved the results.

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ANN-MNIST-Digit-Classifier.git
```

Move into the project:

```bash
cd ANN-MNIST-Digit-Classifier
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📦 Requirements

The project uses:

```text
torch
torchvision
streamlit
streamlit-drawable-canvas
numpy
Pillow
matplotlib
```

These dependencies are listed in:

```text
requirements.txt
```

---

# 📸 Application

The application allows the user to:

1. Draw a handwritten digit.
2. Click **Predict**.
3. Process the drawing.
4. Convert it into an MNIST-compatible format.
5. Pass it through the trained ANN.
6. Display the predicted digit.

---

# 🔮 Future Improvements

- [ ] Add prediction confidence
- [ ] Display probability for all 10 digits
- [ ] Improve the Clear button
- [ ] Add confusion matrix
- [ ] Add training/validation loss graphs
- [ ] Add training/validation accuracy graphs
- [ ] Improve UI design
- [ ] Add model performance statistics
- [ ] Deploy the Streamlit application
- [ ] Experiment with different ANN architectures
- [ ] Compare ANN with CNN
- [ ] Improve preprocessing for different handwriting styles

---

# 📚 What I Would Try Next

The next logical step after this project would be to build a **CNN-based MNIST classifier**.

The ANN treats the image mainly as a flat vector:

```text
28 × 28
   ↓
  784
   ↓
  ANN
```

A CNN can preserve the spatial structure of the image:

```text
28 × 28
   ↓
Convolution
   ↓
Pooling
   ↓
Convolution
   ↓
Pooling
   ↓
Fully Connected Layers
   ↓
Prediction
```

This would allow a comparison between:

```text
ANN vs CNN
```

for image classification.

---

# 📈 Project Learning Pipeline

```text
Python
  ↓
PyTorch Basics
  ↓
MNIST Dataset
  ↓
DataLoader
  ↓
ANN Architecture
  ↓
Training
  ↓
Evaluation
  ↓
Model Saving
  ↓
Image Preprocessing
  ↓
Streamlit
  ↓
Model Deployment
```

---

# 🎯 Project Goal

The main purpose of this project was not just to build a digit classifier.

It was to understand how a **deep learning model moves from training code to a usable application**.

```text
Dataset
   ↓
Model
   ↓
Training
   ↓
Evaluation
   ↓
Saved Model
   ↓
Application
   ↓
Real-world Input
   ↓
Prediction
```

This project helped me understand that building an ML application involves much more than simply training a model.

---

# 👨‍💻 Author

**Aaryan Shelar**

B.Tech CSE — AI & ML

---

⭐ If you found this project useful, consider giving the repository a star!
