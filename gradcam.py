"""
Grad-CAM utility for the Building Crack Detection models (MobileNetV2 / ResNet50).

Both models are flattened Keras functional models with a single sigmoid
output neuron (probability of "Non_Crack"). Grad-CAM is computed by
back-propagating that single output score to the last convolutional layer.
"""

import numpy as np
import tensorflow as tf
import cv2


# Best last-conv-layer target for each architecture (found by inspecting
# the saved .keras models). If you retrain and layer names change, update
# these two constants.
LAST_CONV_LAYER = {
    "MobileNetV2": "out_relu",
    "ResNet50": "conv5_block3_out",
}


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    img_array : preprocessed, batched input, shape (1, H, W, 3)
    model     : the full loaded tf.keras Model
    last_conv_layer_name : name of the conv layer to explain

    Returns a heatmap (H_conv, W_conv) normalized to [0, 1] as a numpy array.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array], training=False)
        # Single sigmoid unit -> the score itself is what we explain.
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # Channel-wise importance weights: global-average-pool the gradients.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # drop batch dim -> (h, w, channels)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)  # ReLU
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        max_val = 1e-8
    heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap(heatmap, original_image_rgb, alpha=0.45):
    """
    heatmap : 2D numpy array in [0, 1], as returned by make_gradcam_heatmap
    original_image_rgb : numpy array (H, W, 3), RGB, uint8 (the ORIGINAL
                          resolution image the user uploaded, not the 224x224
                          model input, so the overlay looks crisp)
    alpha : blend strength of the heatmap

    Returns an RGB uint8 numpy array of the same size as original_image_rgb.
    """
    h, w = original_image_rgb.shape[:2]

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original_image_rgb.astype(np.uint8), 1 - alpha,
        heatmap_color, alpha,
        0,
    )
    return overlay


def generate_gradcam_overlay(model, model_name, preprocessed_input, original_image_rgb):
    """
    Convenience wrapper: does the full pipeline in one call.

    model               : loaded tf.keras Model (mobilenet or resnet)
    model_name          : "MobileNetV2" or "ResNet50" (selects the conv layer)
    preprocessed_input  : batched, preprocessed model input, shape (1, 224, 224, 3)
    original_image_rgb  : the original uploaded image as an RGB numpy array,
                           at whatever resolution you want the overlay drawn at

    Returns: overlay (RGB uint8 numpy array), heatmap (raw 2D array in [0,1])
    """
    layer_name = LAST_CONV_LAYER[model_name]
    heatmap = make_gradcam_heatmap(preprocessed_input, model, layer_name)
    overlay = overlay_heatmap(heatmap, original_image_rgb)
    return overlay, heatmap
