import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Object Recognition",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Object Recognition using Keras")
st.write("Upload an image to identify the object.")

# ----------------------------
# Load Model
# ----------------------------

def load_my_model():
    model = load_model("object_model.keras", compile=False)
    return model

model = load_my_model()

# ----------------------------
# Load Labels
# ----------------------------
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ----------------------------
# Get Input Size from Model
# ----------------------------
input_shape = model.input_shape
img_height = input_shape[1]
img_width = input_shape[2]

# ----------------------------
# Image Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize image
    image_resized = image.resize((img_width, img_height))

    # Convert to array
    img_array = np.asarray(image_resized).astype(np.float32)

    # Normalize
    img_array = (img_array / 127.5) - 1

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    index = np.argmax(prediction)

    confidence = prediction[0][index] * 100

    st.subheader("Prediction")

    st.success(f"Object : {class_names[index]}")
    st.info(f"Confidence : {confidence:.2f}%")

    st.subheader("Prediction Scores")

    for i, score in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {score*100:.2f}%")