import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
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


def analyze_crack(pil_image):
    """Segment the largest crack in the image and measure its area, length, and width."""
    image_rgb = np.array(pil_image)

    # Convert to Gray
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu Threshold
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological Filtering
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    crack_mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Largest Crack Extraction
    contours, _ = cv2.findContours(
        crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_mask = np.zeros_like(crack_mask)
    sketch = image_rgb.copy()
    crack_area = 0
    length_px = 0.0
    width_px = 0.0

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        crack_area = cv2.countNonZero(largest_mask)

        # Rotated bounding box gives an estimate of crack length (longer side)
        # and width (shorter side), both in pixels
        rect = cv2.minAreaRect(largest_contour)
        (rw, rh) = rect[1]
        length_px = float(max(rw, rh))
        width_px = float(min(rw, rh))

        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Sketch: crack outline in red, bounding box in green
        cv2.drawContours(sketch, [largest_contour], -1, (255, 0, 0), 2)
        cv2.drawContours(sketch, [box], 0, (0, 255, 0), 2)

    total_area = crack_mask.shape[0] * crack_mask.shape[1]
    crack_percentage = (crack_area / total_area) * 100 if total_area else 0.0

    return {
        "crack_mask": crack_mask,
        "largest_mask": largest_mask,
        "sketch": sketch,
        "crack_area": crack_area,
        "total_area": total_area,
        "crack_percentage": crack_percentage,
        "length_px": length_px,
        "width_px": width_px,
        "has_crack": bool(contours),
    }


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
    base_img_array = np.array(img).astype("float32")

    def run_prediction(model, preprocess_input):
        img_array = preprocess_input(base_img_array.copy())
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)
        probability = float(prediction[0][0])
        if probability > 0.5:
            return "Non_Crack", probability
        else:
            return "Crack", 1 - probability

    st.markdown("### 🔍 Results")
    cols = st.columns(len(available_models))

    for col, model_name in zip(cols, available_models):
        with col:
            st.markdown(f"**{model_name}**")
            if model_name == "MobileNetV2":
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                predicted_class, confidence = run_prediction(mobilenet_model, preprocess_input)
            else:
                from tensorflow.keras.applications.resnet50 import preprocess_input
                predicted_class, confidence = run_prediction(resnet_model, preprocess_input)

            st.success(f"Prediction: **{predicted_class}**")
            st.info(f"Confidence: **{confidence * 100:.2f}%**")
            st.progress(confidence)

    # --- Crack Measurement (Area, Length, Width, Sketch) ---
    st.markdown("---")
    st.markdown("### 📏 Crack Analysis")

    analysis = analyze_crack(image)

    if not analysis["has_crack"]:
        st.warning("No distinct crack contour could be detected in this image for measurement.")
    else:
        sketch_col, mask_col, largest_col = st.columns(3)

        with sketch_col:
            st.image(analysis["sketch"], caption="Crack Sketch (outline + bounding box)", use_container_width=True)

        with mask_col:
            st.image(analysis["crack_mask"], caption="Crack Mask", use_container_width=True)

        with largest_col:
            st.image(analysis["largest_mask"], caption="Largest Crack Region", use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Crack Area", f"{analysis['crack_area']:.0f} px²")
        m2.metric("Est. Length", f"{analysis['length_px']:.1f} px")
        m3.metric("Est. Width", f"{analysis['width_px']:.1f} px")

        st.info(f"Crack Percentage of Image: **{analysis['crack_percentage']:.2f}%**")

        st.caption(
            "Length and width are pixel-based estimates from the crack's rotated bounding box. "
            "For real-world units (mm/cm), a reference object of known size in the photo is needed for calibration."
        )

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("**Intern:** Sadiqul Islam")
st.markdown("**Mentor:** Debabrat Bharali, Asst. Prof, CSE (AI & DS), Department of Engineering & Technology")
st.markdown(
    "Developed using **TensorFlow**, **MobileNetV2**, **ResNet50**, and **Streamlit**."
)
