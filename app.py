import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Page Configuration
st.set_page_config(
    page_title="Building Crack Detection",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Building Crack Detection")
st.write("Upload a building image to detect whether it contains a crack.")

import os

# Load Models
@st.cache_resource
def load_models():
    mobilenet_path = "mobilnetv2_model.keras"
    resnet_path = "models/resnet50_model.keras"

    mobilenet = tf.keras.models.load_model(mobilenet_path) if os.path.exists(mobilenet_path) else None
    resnet = tf.keras.models.load_model(resnet_path) if os.path.exists(resnet_path) else None
    return mobilenet, resnet

mobilenet_model, resnet_model = load_models()

available_models = []
if mobilenet_model is not None:
    available_models.append("MobileNetV2")
if resnet_model is not None:
    available_models.append("ResNet50")

if not available_models:
    st.error("No model files were found. Please add mobilnetv2_model.keras (and optionally models/resnet50_model.keras) to the app folder.")
    st.stop()


# Class Labels

class_names = ["Crack", "Non_Crack"]

# Model Selection
selected_model = st.selectbox(
    "Choose Model",
    available_models
)

# Image Input Method

st.header("📤 Upload Building Image")

input_method = st.radio(
    "Choose Image Input Method",
    ("Browse Files", "Camera"),
    horizontal=True
)

image = None

# Browse Files (Supports Drag & Drop)

if input_method == "Browse Files":

    uploaded_file = st.file_uploader(
        "Drag & Drop or Browse Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# Camera Capture
elif input_method == "Camera":

    camera_image = st.camera_input("Capture Building Image")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# Prediction
if image is not None:

    st.image(
        image,
        caption="Selected Image",
        use_container_width=True
    )

    # Resize image
    img = image.resize((224, 224))

    img_array = np.array(img).astype("float32")

    # Select model and preprocessing
    if selected_model == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        model = mobilenet_model
    else:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        model = resnet_model

    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    probability = float(prediction[0][0])

    if probability > 0.5:
        predicted_class = "Non_Crack"
        confidence = probability
    else:
        predicted_class = "Crack"
        confidence = 1 - probability

    # Display Results
    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")
    st.progress(confidence)

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("**Intern:** Sadiqul Islam MCA Arunachal University of Studies")
st.markdown("**Mentor:** Debabrat Bharali, Asst. Prof, CSE (AI & DS), Department of Engineering & Technology, USTM")
st.markdown(
    "Developed using **TensorFlow**, **MobileNetV2**, **ResNet50**, and **Streamlit**."
)
