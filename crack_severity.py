"""
Crack severity assessment.

Pipeline (Grad-CAM is used ONLY to find where the crack is, not for the
measurements themselves):

1. Take the Grad-CAM heatmap already computed for the "Crack" prediction and
   threshold it to get a Region Of Interest (ROI) -- the area the model
   actually attended to.
2. Run classical crack segmentation (grayscale -> blur -> Otsu threshold ->
   morphological cleanup), but zero out everything OUTSIDE the ROI first.
   This stops background clutter / shadows / edges elsewhere in the photo
   from being counted as "crack" -- only pixels inside the Grad-CAM
   attention region can end up in the crack mask.
3. From the resulting crack mask, estimate:
       - area (pixels + % of image)
       - length (approx, from contour perimeter)
       - width  (approx average, area / length)
4. Classify severity as Low / Moderate / High from area% and width.

Only call assess_crack_severity() when the model's prediction is "Crack".
For "Non_Crack" predictions there is nothing to measure.
"""

import numpy as np
import cv2


# --- Tunable thresholds ------------------------------------------------------

# Grad-CAM activation threshold used to build the ROI mask. Pixels with
# heatmap intensity below this fraction of the max activation are treated as
# "not attended to" and excluded from crack segmentation.
GRADCAM_ATTENTION_THRESHOLD = 0.45

# Severity thresholds. These are reasonable starting defaults, NOT physically
# calibrated values -- a plain photo has no real-world scale reference, so
# "width in pixels" only makes sense relative to a fixed image size (assumes
# the 224x224 model-input resolution). Recalibrate against your own labeled
# validation images if the severity labels don't match visual expectation.
AREA_PERCENT_THRESHOLDS = {"low": 1.0, "moderate": 5.0}   # % of image area
WIDTH_PX_THRESHOLDS = {"low": 4.0, "moderate": 10.0}      # avg width in px

SEVERITY_COLORS = {
    "Low": "🟢",
    "Moderate": "🟠",
    "High": "🔴",
}

# ------------------------------------------------------------------------------


def gradcam_to_roi_mask(heatmap, image_shape, threshold=GRADCAM_ATTENTION_THRESHOLD):
    """
    heatmap      : 2D array in [0,1], from gradcam.make_gradcam_heatmap
    image_shape  : (H, W, ...) of the target image the mask should match
    Returns a boolean mask (H, W), True where Grad-CAM attention is high.
    """
    h, w = image_shape[:2]
    heatmap_resized = cv2.resize(heatmap.astype("float32"), (w, h))
    return heatmap_resized >= threshold


def segment_crack_in_roi(original_image_rgb, roi_mask):
    """
    Classical Otsu-based crack segmentation, restricted to the Grad-CAM ROI.

    Returns: crack_mask (uint8, 0/255), largest_contour (or None if nothing found)
    """
    gray = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Restrict to the Grad-CAM attention region ONLY -- this is the key step
    # that makes "Grad-CAM highlights only the crack area" true: anything
    # outside the model's attention can never survive into the crack mask.
    binary[~roi_mask] = 0

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    crack_mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

    return crack_mask, largest_contour


def compute_crack_metrics(crack_mask, largest_contour):
    """
    Returns dict: area_px, area_percent, length_px, width_px (all 0 if no
    crack contour was found inside the ROI).
    """
    total_area = crack_mask.shape[0] * crack_mask.shape[1]

    if largest_contour is None or cv2.contourArea(largest_contour) == 0:
        return {"area_px": 0.0, "area_percent": 0.0, "length_px": 0.0, "width_px": 0.0}

    area_px = cv2.contourArea(largest_contour)
    area_percent = (area_px / total_area) * 100

    # A thin, elongated blob's perimeter is roughly twice its length (once
    # along each long edge) -> length ≈ perimeter / 2. This holds reasonably
    # well for crack-like shapes; it will overestimate length for blobby /
    # rounded regions, which is fine here since those get filtered out by the
    # Grad-CAM ROI + Otsu step in the first place.
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
    """
    Combines area% and width into one Low / Moderate / High label, taking
    the worse (higher) of the two individual assessments.
    """
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


def assess_crack_severity(original_image_rgb, heatmap):
    """
    Full pipeline: Grad-CAM heatmap -> ROI -> crack segmentation -> metrics
    -> severity label. Call only when the model predicted "Crack".

    original_image_rgb : RGB uint8 array (the image metrics are computed
                          against, e.g. the 224x224 model input)
    heatmap             : 2D Grad-CAM heatmap in [0,1] (from gradcam.py)

    Returns dict with: severity, metrics, crack_mask, contour_overlay, roi_mask
    """
    roi_mask = gradcam_to_roi_mask(heatmap, original_image_rgb.shape)
    crack_mask, largest_contour = segment_crack_in_roi(original_image_rgb, roi_mask)
    metrics = compute_crack_metrics(crack_mask, largest_contour)
    severity = classify_severity(metrics["area_percent"], metrics["width_px"])
    contour_overlay = draw_crack_overlay(original_image_rgb, largest_contour)

    return {
        "severity": severity,
        "metrics": metrics,
        "crack_mask": crack_mask,
        "contour_overlay": contour_overlay,
        "roi_mask": roi_mask,
    }
