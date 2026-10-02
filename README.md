# Derm-AI — Skin Lesion Classification

A Streamlit web application that classifies skin lesion images into **five**
categories using a fine-tuned **MobileNetV2** model.

## How it works

1. The user uploads a skin image.
2. The image is resized to **224 x 224 px** (the model input size).
3. The **same preprocessing used during training** is applied
   (`keras.applications.mobilenet_v2.preprocess_input`, pixels scaled from
   `[0, 255]` to `[-1, 1]`).
4. The model predicts a probability for each of the five classes.
5. The app displays the **predicted class**, the **confidence score**, and the
   full **probability distribution** for all five classes.
6. A **60% confidence threshold** is applied. Below this threshold the app
   shows an uncertainty message and recommends consulting a dermatologist.

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit application (UI + preprocessing + inference). |
| `mobilenet_v2_optimized.h5` | Trained MobileNetV2 model (HDF5 / Keras 3). |
| `requirements.txt` | Python dependencies. |
| `class_names.json` *(optional)* | Overrides the ordered class labels used in the UI. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the URL printed in the terminal (by default
<http://localhost:8501>).

## Class labels

The label order must match the order used when the model was trained. By
default the app uses:

```
0: Actinic keratosis
1: Basal cell carcinoma
2: Benign keratosis
3: Melanoma
4: Melanocytic nevus
```

To override them without editing code, create `class_names.json` next to
`app.py`:

```json
["Actinic keratosis", "Basal cell carcinoma", "Benign keratosis",
 "Melanoma", "Melanocytic nevus"]
```

## Disclaimer

This project is for **educational / research purposes only**. It does **not**
provide a medical diagnosis. Always consult a qualified dermatologist about
skin concerns.
