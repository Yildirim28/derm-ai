import json
import textwrap
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess_input

DEFAULT_CLASS_NAMES = ["Acne", "Eczema", "Melanoma", "Normal Skin", "Psoriasis"]
CONFIDENCE_THRESHOLD = 0.60

MODEL_REGISTRY = {
    "mobilenet_v2_optimized.h5": {
        "label": "MobileNetV2",
        "note": "Lightweight and fast",
        "preprocess": mv2_preprocess_input,
    },
    "efficientnet_b4_baseline.h5": {
        "label": "EfficientNetB4",
        "note": "Deeper, higher accuracy",
        "preprocess": eff_preprocess_input,
    },
}

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


def html(markup):
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


html(
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

    /* Sidebar: force the light surface so hardcoded text colors stay readable */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fbfcff 0%, #f1f3fd 100%);
        border-right: 1px solid rgba(99,102,241,0.16);
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: #1e2545; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] .stCaption { color: #6b7286; }
    [data-testid="stSidebar"] hr { border-color: rgba(99,102,241,0.18); }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.7rem; }
    [data-testid="stSidebar"] [data-testid="stElementContainer"] { margin-bottom: 0; }

    .brand { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1rem; }
    .brand .logo {
        width: 44px; height: 44px; flex: 0 0 44px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center; font-size: 1.25rem;
        background: linear-gradient(135deg, #4f46e5, #8b5cf6 55%, #ec4899);
        box-shadow: 0 12px 24px -12px rgba(99,102,241,0.95);
    }
    .brand .txt .name { font-size: 1.1rem; font-weight: 800; color: #1e2545; line-height: 1.15; }
    .brand .txt .tag { font-size: 0.74rem; color: #6b7286; }

    .model-chip {
        display: flex; align-items: center; justify-content: space-between; gap: 0.6rem;
        background: #ffffff; border: 1px solid rgba(99,102,241,0.22); border-radius: 14px;
        padding: 0.7rem 0.85rem; margin-bottom: 0.9rem;
        box-shadow: 0 10px 24px -18px rgba(30,41,90,0.7);
    }
    .model-chip .lbl { font-size: 0.64rem; letter-spacing: 0.12em; text-transform: uppercase; color: #8b93a7; font-weight: 700; }
    .model-chip .nm { font-size: 0.95rem; font-weight: 800; color: #4f46e5; }

    [data-testid="stSidebar"] .side-label {
        font-size: 0.66rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase;
        color: #8b93a7; margin: 0.2rem 0 0.55rem;
    }

    .side-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.9rem; }
    .side-grid .info-tile { padding: 0.6rem 0.7rem; }
    .side-grid .info-tile .v { font-size: 0.92rem; }

    .cond {
        display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0.7rem;
        border-radius: 12px; background: rgba(255,255,255,0.85);
        border: 1px solid rgba(99,102,241,0.12); margin-bottom: 0.4rem;
        box-shadow: 0 6px 14px -12px rgba(30,41,90,0.75);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .cond:hover { transform: translateX(2px); box-shadow: 0 8px 16px -12px rgba(30,41,90,0.85); }
    .cond .dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 9px; }
    .cond .txt { font-size: 0.86rem; font-weight: 600; color: #2b3350; }

    [data-testid="stSidebar"] .side-note {
        background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.32);
        border-radius: 14px; padding: 0.65rem 0.8rem; font-size: 0.78rem; color: #92400e;
    }
    [data-testid="stSidebar"] .side-note strong { color: #78350f; }

    .advice {
        display: flex; gap: 0.7rem; align-items: flex-start; text-align: left;
        margin-top: 1.05rem; padding: 0.85rem 1rem; border-radius: 16px;
        background: rgba(14,165,233,0.10); border: 1px solid rgba(14,165,233,0.32);
        color: #075985; font-size: 0.87rem; line-height: 1.45;
    }
    .advice .ico { font-size: 1.1rem; line-height: 1.2; flex: 0 0 auto; }
    .advice strong { color: #0c4a6e; }

    .verdict {
        border-radius: 16px; padding: 0.95rem 1.15rem; margin-bottom: 1rem; font-size: 0.95rem;
    }
    .verdict.ok { background: rgba(16,185,129,0.13); border: 1px solid rgba(16,185,129,0.4); color: #065f46; }
    .verdict.ok strong { color: #064e3b; }
    .verdict.warn { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.45); color: #92400e; }
    .verdict.warn strong { color: #78350f; }

    .cmp-head { margin-bottom: 0.55rem; }
    .cmp-head .cmp-name { font-size: 1.02rem; font-weight: 800; color: #1e2545; }
    .cmp-head .cmp-meta { font-size: 0.74rem; color: #8b93a7; }

    .cmp-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 0.4rem; }
    .cmp-table th {
        text-align: left; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
        color: #8b93a7; font-weight: 700; padding: 0.4rem 0.5rem; border-bottom: 1px solid #e6e9f2;
    }
    .cmp-table td { padding: 0.5rem; font-weight: 600; color: #2b3350; border-bottom: 1px solid #f1f3fa; }

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
    """
)


@st.cache_resource(show_spinner=False)
def load_model(path):
    return tf.keras.models.load_model(path)


def process_image(image, model_input, preprocess_fn):
    size = (model_input, model_input)
    image = image.convert("RGB")
    image = image.resize(size)
    array = tf.keras.preprocessing.image.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    return preprocess_fn(array)


def confidence_band(value):
    if value >= 0.80:
        return "ok", "High confidence"
    if value >= CONFIDENCE_THRESHOLD:
        return "warn", "Moderate confidence"
    return "bad", "Low confidence"


def get_model_input_size(model):
    shape = getattr(model, "input_shape", None)
    return shape[1] if shape and len(shape) == 4 else 224


def predict(path, image):
    spec = AVAILABLE_MODELS[path]
    model = load_model(path)
    size = get_model_input_size(model)
    predictions = model.predict(process_image(image, size, spec["preprocess"]), verbose=0)[0]
    top_idx = int(np.argmax(predictions))
    return {
        "model": path,
        "label": spec["label"],
        "predictions": predictions,
        "top_class": CLASS_NAMES[top_idx],
        "max_prob": float(predictions[top_idx]),
        "input_size": size,
    }


def bars_html(predictions):
    bars = []
    for name, prob in sorted(zip(CLASS_NAMES, predictions), key=lambda x: x[1], reverse=True):
        color = CLASS_META.get(name, {}).get("color", "#6366f1")
        value = prob * 100
        width = max(value, 0.7)
        bars.append(
            '<div class="bar-row">'
            f'<div class="bar-top"><span>{name}</span><span class="val">{value:.2f}%</span></div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{width:.2f}%;background:linear-gradient(90deg,{color}aa,{color});"></div>'
            "</div></div>"
        )
    return "".join(bars)


def gauge_html(max_prob, color):
    band, band_label = confidence_band(max_prob)
    pct = max_prob * 100
    return (
        f'<div class="gauge" style="--p:{pct:.1f}; --g:{color}">'
        '<div class="gauge-inner">'
        f'<div class="pct">{pct:.1f}%</div>'
        '<div class="cap">Confidence</div>'
        "</div></div>"
        f'<div class="chip {band}">{band_label}</div>'
    )


def render_result(predicted_class, max_prob, predictions):
    meta = CLASS_META.get(predicted_class, {"color": "#6366f1", "desc": ""})
    pct = max_prob * 100

    html(
        f"""
        <div class="panel result-card">
            <div class="label">Predicted condition</div>
            <div class="dx" style="color:{meta['color']}">{predicted_class}</div>
            <div class="sub">{meta['desc']}</div>
            {gauge_html(max_prob, meta["color"])}
            <div class="advice">
                <span class="ico">⚠️</span>
                <span>This is an <strong>AI-generated estimate, not a diagnosis</strong>. The model
                can be wrong or miss other conditions. Please consult a certified dermatologist
                or doctor for an accurate medical opinion.</span>
            </div>
        </div>
        """
    )

    html(
        '<div class="panel">'
        "<h3>Probability distribution</h3>"
        f'<p class="muted">Class-wise likelihood produced by the {MODEL_NAME} model.</p>'
        + bars_html(predictions)
        + "</div>"
    )


def render_comparison(results):
    top_classes = {r["top_class"] for r in results}
    agree = len(top_classes) == 1
    shared = top_classes.pop() if agree else None

    if agree:
        banner = (
            f'<div class="verdict ok"><strong>Models agree.</strong> Every model predicted '
            f"<strong>{shared}</strong>.</div>"
        )
    else:
        banner = (
            '<div class="verdict warn"><strong>Models disagree.</strong> They predicted '
            + ", ".join(f"<strong>{c}</strong>" for c in sorted(top_classes))
            + ". Treat this as an uncertain result and consult a doctor.</div>"
        )
    html(banner)

    cols = st.columns(len(results), gap="medium")
    for col, res in zip(cols, results):
        with col:
            meta = CLASS_META.get(res["top_class"], {"color": "#6366f1", "desc": ""})
            html(
                f"""
                <div class="cmp-head">
                    <div class="cmp-name">{res["label"]}</div>
                    <div class="cmp-meta">{res["input_size"]} × {res["input_size"]} input</div>
                </div>
                """
            )
            html(
                f"""
                <div class="panel result-card">
                    <div class="label">Predicted condition</div>
                    <div class="dx" style="color:{meta['color']};font-size:1.45rem;">{res["top_class"]}</div>
                    {gauge_html(res["max_prob"], meta["color"])}
                </div>
                """
            )
            html(
                '<div class="panel">'
                "<h3>Distribution</h3>"
                + bars_html(res["predictions"])
                + "</div>"
            )

    if len(results) > 1:
        html(
            '<div class="panel">'
            "<h3>Head-to-head</h3>"
            '<p class="muted">Top prediction and confidence from each model.</p>'
            '<table class="cmp-table"><thead><tr><th>Model</th><th>Top prediction</th>'
            "<th>Confidence</th></tr></thead><tbody>"
            + "".join(
                f"<tr><td>{r['label']}</td>"
                f'<td style="color:{CLASS_META.get(r["top_class"], {}).get("color", "#6366f1")}">'
                f'{r["top_class"]}</td>'
                f'<td>{r["max_prob"]*100:.2f}%</td></tr>'
                for r in results
            )
            + "</tbody></table>"
            '<div class="advice" style="margin-top:1rem;">'
            '<span class="ico">⚠️</span><span>A disagreement between models means the prediction '
            "is <strong>not reliable</strong>. A higher confidence score does not mean a correct "
            "diagnosis. Please see a certified dermatologist or doctor.</span></div>"
            "</div>"
        )


AVAILABLE_MODELS = {
    path: spec for path, spec in MODEL_REGISTRY.items() if Path(path).exists()
}

if not AVAILABLE_MODELS:
    st.error(
        "No model files were found next to `app.py`. Expected at least one of: "
        + ", ".join(f"`{p}`" for p in MODEL_REGISTRY)
    )
    st.stop()

with st.sidebar:
    html(
        """
        <div class="brand">
            <div class="logo">🩺</div>
            <div class="txt">
                <div class="name">Derm-AI</div>
                <div class="tag">Deep learning skin disease detection</div>
            </div>
        </div>
        """
    )

    mode = st.radio(
        "Mode",
        ["Single model", "Compare models"],
        label_visibility="collapsed",
    )

    html('<div class="side-label">Select a model</div>')
    if mode == "Single model":
        selected_model_path = st.selectbox(
            "CNN Model",
            options=list(AVAILABLE_MODELS),
            format_func=lambda p: AVAILABLE_MODELS[p]["label"],
            index=0,
            label_visibility="collapsed",
        )
        compare_paths = []
    else:
        selected_model_path = None
        compare_paths = st.multiselect(
            "Compare models",
            options=list(AVAILABLE_MODELS),
            default=list(AVAILABLE_MODELS),
            format_func=lambda p: AVAILABLE_MODELS[p]["label"],
            label_visibility="collapsed",
        )
        if len(compare_paths) < 2:
            st.warning("Pick at least two models to compare.")

if mode == "Single model":
    MODEL_NAME = AVAILABLE_MODELS[selected_model_path]["label"]
    model_input = None
    with st.spinner(f"Warming up {MODEL_NAME}..."):
        try:
            model_input = get_model_input_size(load_model(selected_model_path))
        except Exception as exc:
            st.error(f"Unable to load `{selected_model_path}`: {exc}")
            st.stop()
else:
    MODEL_NAME = " + ".join(AVAILABLE_MODELS[p]["label"] for p in compare_paths) or "comparison"
    model_input = None


with st.sidebar:
    if mode == "Single model":
        html(
            f"""
            <div class="model-chip">
                <div>
                    <div class="lbl">Active model</div>
                    <div class="nm">{MODEL_NAME}</div>
                </div>
                <div style="font-size:1.3rem;">🧠</div>
            </div>
            """
        )
        html('<div class="side-label">Model details</div>')
        html(
            f"""
            <div class="side-grid">
                <div class="info-tile"><div class="k">Input</div><div class="v">{model_input} × {model_input}</div></div>
                <div class="info-tile"><div class="k">Classes</div><div class="v">{len(CLASS_NAMES)}</div></div>
                <div class="info-tile"><div class="k">Threshold</div><div class="v">{int(CONFIDENCE_THRESHOLD*100)}%</div></div>
                <div class="info-tile"><div class="k">Framework</div><div class="v">TensorFlow</div></div>
            </div>
            """
        )
    else:
        html(
            f"""
            <div class="model-chip">
                <div>
                    <div class="lbl">Comparing</div>
                    <div class="nm">{len(compare_paths)} models</div>
                </div>
                <div style="font-size:1.3rem;">⚖️</div>
            </div>
            """
        )
        html('<div class="side-label">Models in comparison</div>')
        for path in compare_paths:
            color = CLASS_META.get(CLASS_NAMES[0], {}).get("color", "#6366f1")
            html(
                '<div class="cond">'
                f'<span class="dot" style="background:{color};"></span>'
                f'<span class="txt">{AVAILABLE_MODELS[path]["label"]}</span></div>'
            )

    html('<div class="side-label">Detectable conditions</div>')
    for name in CLASS_NAMES:
        color = CLASS_META.get(name, {}).get("color", "#6366f1")
        html(
            '<div class="cond">'
            f'<span class="dot" style="background:{color};"></span>'
            f'<span class="txt">{name}</span></div>'
        )

    html(
        '<div class="side-note"><strong>Educational project only.</strong> '
        "This tool does not provide a medical diagnosis.</div>"
    )


html(
    f"""
    <div class="hero">
        <span class="badge">AI Dermatology Assistant</span>
        <h1>Skin Disease Detection System</h1>
        <p>Upload a clear, well-lit image of the affected skin area, pick your preferred
        CNN model from the sidebar, and get a fast preliminary assessment.</p>
    </div>
    """
)

html(
    """
    <div class="panel">
        <h3>Step 1 · Upload an image</h3>
        <p class="muted">Supported formats: JPG, JPEG and PNG. Use a close-up, in-focus photo for the best results.</p>
    </div>
    """
)

uploaded_file = st.file_uploader("Upload skin image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

can_analyze = mode == "Single model" or len(compare_paths) >= 2

if uploaded_file is not None and not can_analyze:
    st.info("Select at least two models in the sidebar to run a comparison.")

if uploaded_file is not None and can_analyze:
    image = Image.open(uploaded_file)

    run = st.button("Analyze Image", type="primary", use_container_width=True)

    if run:
        with st.spinner(f"Analyzing with {MODEL_NAME}..."):
            if mode == "Single model":
                result = predict(selected_model_path, image)
                st.session_state["prediction"] = {
                    "name": uploaded_file.name,
                    "mode": "single",
                    "path": selected_model_path,
                    "results": [result],
                }
            else:
                results = [predict(path, image) for path in compare_paths]
                st.session_state["prediction"] = {
                    "name": uploaded_file.name,
                    "mode": "compare",
                    "paths": list(compare_paths),
                    "results": results,
                }

    stored = st.session_state.get("prediction")
    valid = (
        stored
        and stored.get("name") == uploaded_file.name
        and stored.get("mode") == mode
        and (
            (mode == "Single model" and stored.get("path") == selected_model_path)
            or (mode == "Compare models" and stored.get("paths") == list(compare_paths))
        )
    )

    if valid:
        results = stored["results"]

        if mode == "Single model":
            result = results[0]
            predicted_class = result["top_class"]
            max_prob = result["max_prob"]
            predictions = result["predictions"]

            html(
                """
                <div class="panel">
                    <h3>Step 2 · Review the result</h3>
                    <p class="muted">AI output combined with the uploaded image for reference.</p>
                </div>
                """
            )

            col_img, col_res = st.columns([1, 1], gap="large")
            with col_img:
                st.image(image, caption="Uploaded skin image", use_container_width=True)

            with col_res:
                if max_prob < CONFIDENCE_THRESHOLD:
                    html(
                        f"""
                        <div class="alert">
                            <strong>Uncertain prediction.</strong><br>
                            The confidence is <strong>{max_prob*100:.2f}%</strong>, below the
                            <strong>{int(CONFIDENCE_THRESHOLD*100)}%</strong> reliability threshold.
                            Please consult a certified dermatologist for an accurate clinical diagnosis.
                        </div>
                        """
                    )
                render_result(predicted_class, max_prob, predictions)

            html(
                f"""
                <div class="panel">
                    <div class="info-grid">
                        <div class="info-tile"><div class="k">Top prediction</div><div class="v">{predicted_class}</div></div>
                        <div class="info-tile"><div class="k">Confidence</div><div class="v">{max_prob*100:.2f}%</div></div>
                        <div class="info-tile"><div class="k">Model</div><div class="v">{MODEL_NAME}</div></div>
                        <div class="info-tile"><div class="k">Classes</div><div class="v">{len(CLASS_NAMES)}</div></div>
                    </div>
                </div>
                """
            )
        else:
            html(
                """
                <div class="panel">
                    <h3>Step 2 · Compare the results</h3>
                    <p class="muted">Every selected model evaluated the same image, shown side by side.</p>
                </div>
                """
            )

            col_img, col_res = st.columns([1, 2], gap="large")
            with col_img:
                st.image(image, caption="Uploaded skin image", use_container_width=True)
            with col_res:
                html(
                    """
                    <div class="panel">
                        <h3>How to read this</h3>
                        <p class="muted">Compare each model's top prediction and confidence. When the
                        models disagree, the result is unreliable and must be confirmed by a doctor.</p>
                    </div>
                    """
                )

            render_comparison(results)
elif uploaded_file is None:
    html(
        '<p style="text-align:center;color:#8b93a7;font-size:0.9rem;margin-top:1rem;">'
        "No image uploaded yet.</p>"
    )

st.markdown("---")
st.caption(
    "Disclaimer: This tool is a final-year academic project intended for educational purposes only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment."
)
