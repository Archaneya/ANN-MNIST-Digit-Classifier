import streamlit as st
import torch
from streamlit_drawable_canvas import st_canvas
from src.model import ANN
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# -----------------------------
# Page Configuration
# -----------------------------

st.title("🧠 MNIST Digit Classifier")
st.caption("Draw a handwritten digit and let the ANN predict it.")


# -----------------------------
# Load Trained Model
# -----------------------------

@st.cache_resource
def load_model():
    model = ANN(784)

    model.load_state_dict(
        torch.load(
            "mnist_ann.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


# -----------------------------
# Drawing Canvas
# -----------------------------

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


# -----------------------------
# Buttons
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("🔍 Predict")

with col2:
    clear_button = st.button("🗑️ Clear")


# -----------------------------
# Prediction
# -----------------------------

if predict_button:

    if canvas_result.image_data is not None:

        # Get canvas image
        image = canvas_result.image_data.astype(np.uint8)

        # Convert RGBA → grayscale
        image = Image.fromarray(image).convert("L")

        # Convert image to NumPy array
        image_array = np.array(image)

        # Find pixels belonging to the digit
        coords = np.argwhere(image_array > 20)

        if coords.size > 0:

            # Get bounding box
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Crop the digit
            cropped = image.crop(
                (x_min, y_min, x_max + 1, y_max + 1)
            )

            # Resize while maintaining aspect ratio
            cropped.thumbnail((20, 20))

            # Create a 28 × 28 black image
            processed_image = Image.new(
                "L",
                (28, 28),
                0
            )

            # Calculate position to center digit
            x_offset = (28 - cropped.width) // 2
            y_offset = (28 - cropped.height) // 2

            # Paste digit into center
            processed_image.paste(
                cropped,
                (x_offset, y_offset)
            )

            # Show processed image
            st.image(
                processed_image,
                caption="Processed MNIST Input"
            )

            # Convert image to PyTorch tensor
            image_tensor = transforms.ToTensor()(
                processed_image
            )

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

            # Make prediction
            with torch.no_grad():

                output = model(image_tensor)

            # Get predicted digit
            prediction = torch.argmax(
                output,
                dim=1
            ).item()

            # Display prediction
            st.success(
                f"🎯 Prediction: {prediction}"
            )

        else:

            st.warning(
                "Please draw a digit first."
            )