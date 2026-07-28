import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

from gradcam import generate_gradcam_overlay
from crack_severity import assess_crack_severity, SEVERITY_COLORS


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

show_gradcam = st.checkbox("Show Grad-CAM heatmap (highlight what the model focused on)", value=True)

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

    # Keep a clean, unmodified copy of the 224x224 RGB array for Grad-CAM overlay,
    # since preprocess_input() below mutates/rescales pixel values in place.
    img_array_raw = np.array(img).astype("float32")
    original_rgb_for_overlay = np.array(img).astype("uint8")

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

    # Grad-CAM (always computed once a prediction is made -- needed both for
    # the optional visualization below AND for crack severity assessment)
    with st.spinner("Running Grad-CAM..."):
        overlay, heatmap = generate_gradcam_overlay(
            model=model,
            model_name=selected_model,
            preprocessed_input=img_array,
            original_image_rgb=original_rgb_for_overlay,
        )

    if show_gradcam:
        st.subheader("🔍 Grad-CAM: What the model looked at")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_rgb_for_overlay, caption="Input (224x224)", use_container_width=True)
        with col2:
            st.image(overlay, caption=f"Grad-CAM ({selected_model})", use_container_width=True)
        st.caption(
            "Red/yellow regions contributed most to the prediction; "
            "blue regions contributed least."
        )

    # Crack Severity Assessment
    # Grad-CAM is used ONLY to localize the crack area: classical
    # segmentation (Otsu + morphology) runs restricted to the region the
    # model attended to, so background clutter never gets counted as crack.
    if predicted_class == "Crack":
        result = assess_crack_severity(original_rgb_for_overlay, heatmap)
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

        col3, col4 = st.columns(2)
        with col3:
            st.image(overlay, caption="Grad-CAM attention region", use_container_width=True)
        with col4:
            st.image(
                result["contour_overlay"],
                caption="Detected crack boundary (within Grad-CAM region)",
                use_container_width=True,
            )

        st.caption(
            "Length/width/area are pixel-based estimates from the region the "
            "model's Grad-CAM attention highlighted, not physical measurements "
            "(no real-world scale reference exists in a plain photo). "
            "Severity thresholds are heuristic defaults -- recalibrate them in "
            "`crack_severity.py` against your own labeled images if needed."
        )

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("**Intern:** Sadiqul Islam")
st.markdown("**Mentor:** Debabrat Bharali, Asst. Prof, CSE (AI & DS), Department of Engineering & Technology")
st.markdown(
    "Developed using **TensorFlow**, **MobileNetV2**, **ResNet50**, and **Streamlit**."
)
