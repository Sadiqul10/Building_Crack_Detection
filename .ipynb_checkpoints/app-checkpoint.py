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

# Load Models
@st.cache_resource
def load_models():
    mobilenet = tf.keras.models.load_model("models/mobilnetv2_model.keras"")
    resnet = tf.keras.models.load_model("models/resnet50_model.keras")
    return mobilenet, resnet

mobilenet_model, resnet_model = load_models()


# Class Labels

class_names = ["Crack", "Non_Crack"]

# Model Selection
selected_model = st.selectbox(
    "Choose Model",
    ["MobileNetV2", "ResNet50"]
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
st.markdown("**Intern:** Sadiqul Islam")
st.markdown("**Mentor:** Debabrat Bharali, Asst. Prof, CSE (AI & DS), Department of Engineering & Technology")
st.markdown(
    "Developed using **TensorFlow**, **MobileNetV2**, **ResNet50**, and **Streamlit**."
)