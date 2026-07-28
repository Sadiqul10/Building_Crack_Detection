import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image


# ==============================================================================
# Crack Severity Assessment
#
# Classical segmentation (Otsu threshold + morphological cleanup) finds the
# crack region directly in the photo. From the resulting crack mask:
# estimate length, width, area%, then classify Low / Moderate / High severity.
# ==============================================================================

# Severity thresholds -- heuristic starting defaults, NOT physically
# calibrated (a plain photo has no real-world scale reference). Recalibrate
# against your own labeled validation images if needed.
AREA_PERCENT_THRESHOLDS = {"low": 1.0, "moderate": 5.0}   # % of image area
WIDTH_PX_THRESHOLDS = {"low": 4.0, "moderate": 10.0}      # avg width in px

SEVERITY_COLORS = {
    "Low": "🟢",
    "Moderate": "🟠",
    "High": "🔴",
}


def segment_crack(original_image_rgb):
    """
    Classical Otsu-based crack segmentation on the full image.
    Returns: crack_mask (uint8, 0/255), largest_contour (or None)
    """
    gray = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    crack_mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

    return crack_mask, largest_contour


def compute_crack_metrics(crack_mask, largest_contour):
    """Returns dict: area_px, area_percent, length_px, width_px."""
    total_area = crack_mask.shape[0] * crack_mask.shape[1]

    if largest_contour is None or cv2.contourArea(largest_contour) == 0:
        return {"area_px": 0.0, "area_percent": 0.0, "length_px": 0.0, "width_px": 0.0}

    area_px = cv2.contourArea(largest_contour)
    area_percent = (area_px / total_area) * 100

    # A thin, elongated blob's perimeter is roughly twice its length.
    perimeter = cv2.arcLength(largest_contour, closed=True)
    length_px = perimeter / 2.0

    width_px = area_px / length_px if length_px > 0 else 0.0

    return {
        "area_px": float(area_px),
        "area_percent": float(area_percent),
        "length_px": float(length_px),
        "width_px": float(width_px),
    }


def classify_severity(area_percent, width_px):
    """Combines area% and width into Low/Moderate/High, taking the worse level."""
    def level_from_area(a):
        if a < AREA_PERCENT_THRESHOLDS["low"]:
            return 0
        elif a < AREA_PERCENT_THRESHOLDS["moderate"]:
            return 1
        return 2

    def level_from_width(w):
        if w < WIDTH_PX_THRESHOLDS["low"]:
            return 0
        elif w < WIDTH_PX_THRESHOLDS["moderate"]:
            return 1
        return 2

    level = max(level_from_area(area_percent), level_from_width(width_px))
    return ["Low", "Moderate", "High"][level]


def draw_crack_overlay(original_image_rgb, largest_contour):
    """Draws the detected crack contour (green) for visual verification."""
    overlay = original_image_rgb.copy()
    if largest_contour is not None:
        cv2.drawContours(overlay, [largest_contour], -1, (0, 255, 0), 2)
    return overlay


def assess_crack_severity(original_image_rgb):
    """
    Full pipeline: crack segmentation -> metrics -> severity label.
    Call only when the model predicted "Crack".
    """
    crack_mask, largest_contour = segment_crack(original_image_rgb)
    metrics = compute_crack_metrics(crack_mask, largest_contour)
    severity = classify_severity(metrics["area_percent"], metrics["width_px"])
    contour_overlay = draw_crack_overlay(original_image_rgb, largest_contour)

    return {
        "severity": severity,
        "metrics": metrics,
        "crack_mask": crack_mask,
        "contour_overlay": contour_overlay,
    }


# Page Configuration
st.set_page_config(
    page_title="Building Crack Detection",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Building Crack Detection")
st.write("Upload a building image to detect whether it contains a crack.")

# Load Models
# Paths are relative to this script's location (works both locally and on
# Streamlit Community Cloud, where the repo is mounted at /mount/src/<repo>/).
# Do NOT hardcode an absolute local path like "C:\..." -- it won't exist on
# the deployment server.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MOBILENET_PATH = os.path.join(APP_DIR, "models", "mobilnetv2_model.keras")
RESNET_PATH = os.path.join(APP_DIR, "models", "resnet50_model.keras")


@st.cache_resource
def load_models():
    missing = [p for p in (MOBILENET_PATH, RESNET_PATH) if not os.path.exists(p)]
    if missing:
        st.error(
            "Model file(s) not found:\n"
            + "\n".join(f"- {p}" for p in missing)
            + f"\n\nApp directory: {APP_DIR}\n"
            + "Make sure your repo has a `models/` folder next to app.py "
            "containing `mobilnetv2_model.keras` and `resnet50_model.keras`, "
            "and that both files are committed to git (they are NOT ignored "
            "by .gitignore, and not tracked via Git LFS without LFS being "
            "enabled on Streamlit Cloud)."
        )
        st.stop()

    try:
        mobilenet = tf.keras.models.load_model(MOBILENET_PATH)
        resnet = tf.keras.models.load_model(RESNET_PATH)
    except Exception as e:
        st.error(
            f"Failed to load model files: {e}\n\n"
            "This usually means the TensorFlow/Keras version used to save "
            "the models doesn't match the version installed here. Check "
            "requirements.txt pins the same tensorflow version you trained with."
        )
        st.stop()

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

    # Resize image (model input size)
    img = image.resize((224, 224))

    # Keep a clean, unmodified copy of the 224x224 RGB array for crack
    # segmentation, since preprocess_input() below mutates pixel values.
    img_array_raw = np.array(img).astype("float32")
    original_rgb_for_analysis = np.array(img).astype("uint8")

    # Select model and preprocessing
    if selected_model == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        model = mobilenet_model
    else:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        model = resnet_model

    img_array = preprocess_input(img_array_raw.copy())
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

    # Crack Severity Assessment
    if predicted_class == "Crack":
        result = assess_crack_severity(original_rgb_for_analysis)
        severity = result["severity"]
        metrics = result["metrics"]

        st.subheader("📏 Crack Severity Assessment")

        badge = SEVERITY_COLORS.get(severity, "")
        if severity == "High":
            st.error(f"{badge} Severity: **{severity}**")
        elif severity == "Moderate":
            st.warning(f"{badge} Severity: **{severity}**")
        else:
            st.success(f"{badge} Severity: **{severity}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Crack length (approx.)", f"{metrics['length_px']:.1f} px")
        m2.metric("Crack width (approx.)", f"{metrics['width_px']:.1f} px")
        m3.metric("Crack area", f"{metrics['area_percent']:.2f}%")

        st.image(
            result["contour_overlay"],
            caption="Detected crack boundary",
            use_container_width=True,
        )

        st.caption(
            "Length/width/area are pixel-based estimates from the segmented "
            "crack region, not physical measurements (no real-world scale "
            "reference exists in a plain photo). Severity thresholds are "
            "heuristic defaults -- recalibrate the constants at the top of "
            "this file against your own labeled images if needed."
        )

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("**Intern:** Sadiqul Islam")
st.markdown("**Mentor:** Debabrat Bharali, Asst. Prof, CSE (AI & DS), Department of Engineering & Technology")
st.markdown(
    "Developed using **TensorFlow**, **MobileNetV2**, **ResNet50**, and **Streamlit**."
)
