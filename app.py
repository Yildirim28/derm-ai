import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "skin_disease_mobilenetv2.h5"
DEFAULT_CLASS_NAMES = ["Acne", "Eczema", "Melanoma", "Normal Skin", "Psoriasis"]
CONFIDENCE_THRESHOLD = 0.60

CLASS_META = {
    "Acne": {"color": "#6366f1", "desc": "Inflammatory skin condition affecting hair follicles and sebaceous glands."},
    "Eczema": {"color": "#0ea5e9", "desc": "Chronic inflammatory condition causing dry, itchy and irritated skin."},
    "Melanoma": {"color": "#ef4444", "desc": "A serious form of skin cancer arising from pigment-producing cells."},
    "Normal Skin": {"color": "#10b981", "desc": "Healthy skin with no significant lesions detected."},
    "Psoriasis": {"color": "#f59e0b", "desc": "Autoimmune condition producing raised, scaly patches on the skin."},
}


def load_class_names():
    override = Path("class_names.json")
    if override.exists():
        try:
            names = json.loads(override.read_text())
            if isinstance(names, list) and names:
                return [str(n) for n in names]
        except (ValueError, OSError):
            pass
    return DEFAULT_CLASS_NAMES


CLASS_NAMES = load_class_names()

st.set_page_config(
    page_title="Derm-AI | Skin Disease Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(900px 500px at 12% -8%, rgba(99,102,241,0.16), transparent 60%),
            radial-gradient(800px 480px at 92% 4%, rgba(236,72,153,0.14), transparent 60%),
            linear-gradient(180deg, #f7f9ff 0%, #eef1fb 100%);
    }

    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px; }

    .hero { text-align: center; padding: 1.2rem 1rem 0.4rem; }
    .hero .badge {
        display: inline-block; padding: 0.32rem 0.9rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
        color: #6366f1; background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.28);
    }
    .hero h1 {
        font-size: 2.7rem; font-weight: 800; margin: 0.7rem 0 0.35rem; line-height: 1.1;
        background: linear-gradient(95deg, #4f46e5, #8b5cf6 45%, #ec4899);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: #5b6478; font-size: 1.02rem; max-width: 680px; margin: 0 auto; }

    .panel {
        background: rgba(255,255,255,0.82); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.9); border-radius: 20px;
        box-shadow: 0 18px 45px -22px rgba(30,41,90,0.45);
        padding: 1.5rem 1.6rem; margin-bottom: 1.1rem;
    }
    .panel h3 { margin: 0 0 0.25rem; font-size: 1.05rem; font-weight: 700; color: #1e2545; }
    .panel .muted { color: #6b7286; font-size: 0.9rem; margin: 0; }

    .result-card { text-align: center; }
    .result-card .label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #8b93a7;
    }
    .result-card .dx { font-size: 1.9rem; font-weight: 800; margin: 0.15rem 0 0.1rem; }
    .result-card .sub { color: #6b7286; font-size: 0.9rem; margin-bottom: 1rem; }

    .gauge {
        --p: 0;
        width: 156px; height: 156px; margin: 0.4rem auto 0.6rem; border-radius: 50%;
        background: conic-gradient(var(--g) calc(var(--p) * 1%), #e6e9f2 0);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 14px 30px -16px rgba(30,41,90,0.55);
    }
    .gauge-inner {
        width: 120px; height: 120px; border-radius: 50%; background: #ffffff;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .gauge-inner .pct { font-size: 1.7rem; font-weight: 800; color: #1e2545; line-height: 1; }
    .gauge-inner .cap { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: #8b93a7; margin-top: 0.25rem; }

    .chip {
        display: inline-block; margin-top: 0.35rem; padding: 0.35rem 0.85rem; border-radius: 999px;
        font-size: 0.82rem; font-weight: 700;
    }
    .chip.ok { color: #047857; background: rgba(16,185,129,0.14); border: 1px solid rgba(16,185,129,0.35); }
    .chip.warn { color: #b45309; background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); }
    .chip.bad { color: #b91c1c; background: rgba(239,68,68,0.14); border: 1px solid rgba(239,68,68,0.4); }

    .alert {
        border-radius: 16px; padding: 1rem 1.15rem; margin-bottom: 0.9rem;
        border: 1px solid rgba(245,158,11,0.4); background: rgba(254,243,199,0.55);
        color: #92400e; font-size: 0.94rem;
    }
    .alert strong { color: #78350f; }

    .bar-row { margin: 0.65rem 0; }
    .bar-top { display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; color: #2b3350; margin-bottom: 0.3rem; }
    .bar-top .val { color: #6b7286; font-variant-numeric: tabular-nums; }
    .bar-track { height: 10px; border-radius: 999px; background: #e6e9f2; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }

    .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.7rem; margin-top: 0.4rem; }
    .info-tile { background: #ffffff; border: 1px solid #eceff6; border-radius: 14px; padding: 0.75rem 0.9rem; }
    .info-tile .k { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: #8b93a7; font-weight: 700; }
    .info-tile .v { font-size: 1.02rem; font-weight: 700; color: #1e2545; margin-top: 0.15rem; }

    .stButton > button {
        width: 100%; border: none; border-radius: 14px; padding: 0.8rem 1rem;
        font-weight: 700; font-size: 1rem; color: #ffffff;
        background: linear-gradient(95deg, #4f46e5, #8b5cf6 55%, #ec4899);
        box-shadow: 0 16px 30px -14px rgba(99,102,241,0.85);
    }
    .stButton > button:hover { filter: brightness(1.05); }
    .stButton > button:disabled { opacity: 0.55; }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.82); border: 1px dashed rgba(99,102,241,0.45);
        border-radius: 18px; padding: 1rem;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(path):
    return tf.keras.models.load_model(path)


def process_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    array = tf.keras.preprocessing.image.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)


def confidence_band(value):
    if value >= 0.80:
        return "ok", "High confidence"
    if value >= CONFIDENCE_THRESHOLD:
        return "warn", "Moderate confidence"
    return "bad", "Low confidence"


def render_result(predicted_class, max_prob, predictions):
    meta = CLASS_META.get(predicted_class, {"color": "#6366f1", "desc": ""})
    band, band_label = confidence_band(max_prob)
    pct = max_prob * 100

    st.markdown(
        f"""
        <div class="panel result-card">
            <div class="label">Predicted condition</div>
            <div class="dx" style="color:{meta['color']}">{predicted_class}</div>
            <div class="sub">{meta['desc']}</div>
            <div class="gauge" style="--p:{pct:.1f}; --g:{meta['color']}">
                <div class="gauge-inner">
                    <div class="pct">{pct:.1f}%</div>
                    <div class="cap">Confidence</div>
                </div>
            </div>
            <div class="chip {band}">{band_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bars = []
    for name, prob in sorted(zip(CLASS_NAMES, predictions), key=lambda x: x[1], reverse=True):
        color = CLASS_META.get(name, {}).get("color", "#6366f1")
        value = prob * 100
        bars.append(
            f"""
            <div class="bar-row">
                <div class="bar-top"><span>{name}</span><span class="val">{value:.2f}%</span></div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{max(value, 0.7):.2f}%;background:linear-gradient(90deg,{color}aa,{color});"></div>
                </div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="panel">
            <h3>Probability distribution</h3>
            <p class="muted">Class-wise likelihood produced by the MobileNetV2 model.</p>
            {''.join(bars)}
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.spinner("Warming up the neural network..."):
    if not Path(MODEL_PATH).exists():
        st.error(f"Model file `{MODEL_PATH}` was not found. Place it next to `app.py`.")
        st.stop()
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"Unable to load the model: {exc}")
        st.stop()


with st.sidebar:
    st.markdown("### Derm-AI")
    st.caption("Deep learning based skin disease detection.")
    st.markdown(
        """
        <div class="info-grid">
            <div class="info-tile"><div class="k">Architecture</div><div class="v">MobileNetV2</div></div>
            <div class="info-tile"><div class="k">Input size</div><div class="v">224 × 224</div></div>
            <div class="info-tile"><div class="k">Classes</div><div class="v">5</div></div>
            <div class="info-tile"><div class="k">Threshold</div><div class="v">60%</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Detectable conditions")
    for name in CLASS_NAMES:
        color = CLASS_META.get(name, {}).get("color", "#6366f1")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.3rem 0;">'
            f'<span style="width:9px;height:9px;border-radius:50%;background:{color};"></span>'
            f'<span style="font-size:0.9rem;color:#2b3350;">{name}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Educational project only. Not a medical diagnosis.")


st.markdown(
    """
    <div class="hero">
        <span class="badge">AI Dermatology Assistant</span>
        <h1>Skin Disease Detection System</h1>
        <p>Upload a clear, well-lit image of the affected skin area and let our
        MobileNetV2 model provide a fast preliminary assessment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="panel">
        <h3>Step 1 · Upload an image</h3>
        <p class="muted">Supported formats: JPG, JPEG and PNG. Use a close-up, in-focus photo for the best results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload skin image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    run = st.button("Analyze Image", type="primary", use_container_width=True)

    if run:
        with st.spinner("Analyzing image features..."):
            predictions = model.predict(process_image(image), verbose=0)[0]
        st.session_state["prediction"] = {
            "name": uploaded_file.name,
            "predictions": predictions.tolist(),
        }

    stored = st.session_state.get("prediction")

    if stored and stored.get("name") == uploaded_file.name:
        predictions = np.array(stored["predictions"])
        top_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[top_idx]
        max_prob = float(predictions[top_idx])

        st.markdown(
            '<div class="panel"><h3>Step 2 · Review the result</h3>'
            '<p class="muted">AI output combined with the uploaded image for reference.</p></div>',
            unsafe_allow_html=True,
        )

        col_img, col_res = st.columns([1, 1], gap="large")
        with col_img:
            st.image(image, caption="Uploaded skin image", use_container_width=True)

        with col_res:
            if max_prob < CONFIDENCE_THRESHOLD:
                st.markdown(
                    f"""
                    <div class="alert">
                        <strong>Uncertain prediction.</strong><br>
                        The confidence is <strong>{max_prob*100:.2f}%</strong>, below the
                        <strong>{int(CONFIDENCE_THRESHOLD*100)}%</strong> reliability threshold.
                        Please consult a certified dermatologist for an accurate clinical diagnosis.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            render_result(predicted_class, max_prob, predictions)

        st.markdown(
            f"""
            <div class="panel">
                <div class="info-grid">
                    <div class="info-tile"><div class="k">Top prediction</div><div class="v">{predicted_class}</div></div>
                    <div class="info-tile"><div class="k">Confidence</div><div class="v">{max_prob*100:.2f}%</div></div>
                    <div class="info-tile"><div class="k">Model</div><div class="v">MobileNetV2</div></div>
                    <div class="info-tile"><div class="k">Classes</div><div class="v">{len(CLASS_NAMES)}</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<p style="text-align:center;color:#8b93a7;font-size:0.9rem;margin-top:1rem;">'
        "No image uploaded yet.</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "Disclaimer: This tool is a final-year academic project intended for educational purposes only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment."
)
