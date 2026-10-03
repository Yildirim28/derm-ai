"""
Skin Lesion Classification - Streamlit Web Application
======================================================

This app serves the MobileNetV2 model that was fine-tuned during the
optimization stage and saved as ``mobilenet_v2_optimized.h5``.

Workflow
--------
1. The user uploads a skin image.
2. The image is resized to 224 x 224 (the model input size, taken from the
   saved model's InputLayer: ``[None, 224, 224, 3]``).
3. The *exact* preprocessing used during training is applied:
   ``keras.applications.mobilenet_v2.preprocess_input``
   (i.e. pixels scaled from [0, 255] to [-1, 1]).
4. The model predicts a probability for each of the five classes.
5. The app displays the predicted class, the confidence score and the full
   probability distribution for all five classes.
6. A 60% confidence threshold is used. If the top confidence is below the
   threshold, an uncertainty message is shown and a dermatologist
   consultation is recommended.
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_v2_optimized.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

# Input size the model was trained on (from the saved InputLayer).
IMG_SIZE = (224, 224)

# Below this top-class probability the prediction is treated as uncertain.
CONFIDENCE_THRESHOLD = 0.60

# The order MUST match the label order used when the model was trained
# (i.e. the sorted class order of the training generator / label encoder).
#
# You can override this WITHOUT editing code by creating ``class_names.json``
# next to this file, e.g.:
#   ["Actinic keratosis", "Basal cell carcinoma", "Benign keratosis",
#    "Melanoma", "Melanocytic nevus"]
CLASS_NAMES = ['Acne', 'Eczema', 'Melanoma', 'Normal Skin', 'Psoriasis']


# --------------------------------------------------------------------------- #
# Resource loaders (cached so they run only once)
# --------------------------------------------------------------------------- #
def load_class_names():
    """Return the ordered class labels (from JSON if present, else default)."""
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as handle:
                names = json.load(handle)
            if isinstance(names, list) and len(names) == len(DEFAULT_CLASS_NAMES):
                return [str(name) for name in names]
        except (ValueError, OSError):
            # Fall back to the defaults if the file is unreadable/invalid.
            pass
    return DEFAULT_CLASS_NAMES


@st.cache_resource(show_spinner="Loading the trained model ...")
def load_model():
    """Load the HDF5 Keras model once and cache it.

    ``compile=False`` is used because only inference is required; this avoids
    depending on the saved optimizer / training configuration.
    """
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# --------------------------------------------------------------------------- #
# Preprocessing + inference
# --------------------------------------------------------------------------- #
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and preprocess a PIL image into a (1, 224, 224, 3) batch.

    Matches the training-time pipeline: RGB -> 224x224 -> float32 in [0, 255]
    -> MobileNetV2 ``preprocess_input`` (scaled to [-1, 1]).
    """
    image = image.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)          # [0, 255]
    array = preprocess_input(array)                       # [-1, 1]
    return np.expand_dims(array, axis=0)                  # (1, 224, 224, 3)


def predict_probabilities(model, image: Image.Image) -> np.ndarray:
    """Return the softmax probability vector for a single image."""
    batch = preprocess_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    return np.asarray(probabilities, dtype=np.float64)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def render_prediction(probabilities: np.ndarray, class_names):
    """Render the predicted class, confidence and probability distribution."""
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_class = class_names[predicted_index]

    st.subheader("Prediction")

    if confidence < CONFIDENCE_THRESHOLD:
        st.error(
            f"**Uncertain result.** The model's best guess is "
            f"*{predicted_class}* with only **{confidence:.1%}** confidence, "
            f"which is below the {CONFIDENCE_THRESHOLD:.0%} threshold."
        )
        st.warning(
            "The model is not confident enough to give a reliable answer for "
            "this image. **Please consult a dermatologist** for a proper "
            "clinical assessment. This tool is for educational purposes only "
            "and must not be used as a medical diagnosis."
        )
    else:
        st.success(
            f"Predicted class: **{predicted_class}** "
            f"(confidence **{confidence:.1%}**)"
        )

    metric_cols = st.columns(2)
    metric_cols[0].metric("Predicted class", predicted_class)
    metric_cols[1].metric("Confidence score", f"{confidence:.1%}")

    st.subheader("Probability distribution (all 5 classes)")
    distribution = pd.DataFrame(
        {"Probability": probabilities}, index=class_names
    )
    st.bar_chart(distribution)
    st.dataframe(distribution.style.format("{:.2%}"))


def main():
    st.set_page_config(
        page_title="Skin Lesion Classifier",
        page_icon="\U0001F52C",
        layout="wide",
    )

    class_names = load_class_names()

    st.title("\U0001F52C Skin Lesion Classification")
    st.write(
        "Upload a skin image and the fine-tuned **MobileNetV2** model will "
        "classify it into one of the five lesion categories and report the "
        "confidence of its prediction."
    )

    # ----- Sidebar ---------------------------------------------------------- #
    with st.sidebar:
        st.header("About")
        st.markdown(
            f"- **Model:** MobileNetV2 (fine-tuned)\n"
            f"- **Input size:** {IMG_SIZE[0]} x {IMG_SIZE[1]} px\n"
            f"- **Classes:** {len(class_names)}\n"
            f"- **Confidence threshold:** {CONFIDENCE_THRESHOLD:.0%}"
        )
        st.markdown("**Classes**")
        for index, name in enumerate(class_names):
            st.markdown(f"{index}: {name}")

        st.divider()
        st.caption(
            "For educational/research use only. This application does not "
            "provide a medical diagnosis. Always consult a qualified "
            "dermatologist about skin concerns."
        )

    # ----- Model check ------------------------------------------------------ #
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found: `{MODEL_PATH}`. "
            "Place `mobilenet_v2_optimized.h5` next to `app.py`."
        )
        st.stop()

    model = load_model()

    # ----- Upload ----------------------------------------------------------- #
    uploaded_file = st.file_uploader(
        "Choose a skin image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, JPEG, PNG, BMP, WEBP.",
    )

    if uploaded_file is None:
        st.info("Upload an image to get a prediction.")
        return

    image = Image.open(uploaded_file)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Uploaded image")
        st.image(image, caption="Resized to 224 x 224 before prediction",
                 width=300)

    with right:
        with st.spinner("Running inference ..."):
            probabilities = predict_probabilities(model, image)
        render_prediction(probabilities, class_names)


if __name__ == "__main__":
    main()

