import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. PAGE CONFIG & MODERN STYLING
# ==========================================
st.set_page_config(
    page_title="DermAI — Skin Disease Classifier",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for modern glassmorphism & clean UI
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2b6cb0, #4fd1c5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR TAB: PROJECT & SYSTEM DETAILS
# ==========================================
with st.sidebar:
    st.markdown("### 🧬 **DermAI**")
    st.caption("Advanced Deep Learning Healthcare System")
    st.markdown("---")
    
    st.markdown("#### 📌 About the Project")
    st.write(
        "**DermAI** is an intelligent web application designed as a "
        "Final Year Design Project (FYDP) to provide fast, reliable preliminary "
        "skin disease classifications using computer vision."
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ Model Specifications")
    st.markdown("""
    * **Architecture:** MobileNetV2 *(Optimized)*
    * **Input Dimension:** 224 × 224 px
    * **Target Classes:** 5 Categories
    * **Safety Threshold:** 60% Confidence
    """)
    
    st.markdown("---")
    st.markdown("#### 🏫 Institution & Context")
    st.markdown("""
    * **Department:** Computer Science & Engineering
    * **Affiliation:** United International University (UIU)
    """)
    
    st.markdown("---")
    st.caption("© 2026 DermAI Research Initiative")

# ==========================================
# 3. MAIN HEADER SECTION
# ==========================================
st.markdown('<p class="main-title">🩺 DermAI: Skin Disease Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload a clear dermatological image below to receive an instant, AI-driven preliminary assessment.</p>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 4. LOAD MODEL (Cached for Performance)
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mobilenet_v2_optimized.h5')

with st.spinner("🔄 Loading Optimized AI Engine... Please wait."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        st.info("💡 **Tip:** Ensure `mobilenet_v2_optimized.h5` is placed in the exact same directory as `app.py`.")
        st.stop()

CLASS_NAMES = ['Acne', 'Eczema', 'Melanoma', 'Normal Skin', 'Psoriasis']

# ==========================================
# 5. IMAGE PREPROCESSING FUNCTION
# ==========================================
def process_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# ==========================================
# 6. DUAL-COLUMN UI: UPLOAD & PREDICTION
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### 📤 Step 1: Upload Image")
    uploaded_file = st.file_uploader("Choose a skin lesion photo...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Skin Sample', use_column_width=True)

with col2:
    st.markdown("#### 🔍 Step 2: Diagnostic Analysis")
    if uploaded_file is not None:
        if st.button("Run AI Classification", type="primary", use_container_width=True):
            with st.spinner("Analyzing patterns and feature maps..."):
                processed_img = process_image(image)
                predictions = model.predict(processed_img)[0]
                
                max_prob = np.max(predictions)
                predicted_class = CLASS_NAMES[np.argmax(predictions)]
                
                st.markdown("---")
                
                # Uncertainty Handling (60% Threshold)
                if max_prob < 0.60:
                    st.warning("⚠️ **Uncertain Prediction Detected**")
                    st.write(f"Model confidence stands at **{max_prob*100:.2f}%**, which falls below our strict **60% reliability threshold**.")
                    st.info("💡 **Recommendation:** Due to low certainty, please consult a certified dermatologist for professional diagnosis.")
                else:
                    st.success(f"### Predicted Condition: **{predicted_class}**")
                    st.metric(label="Model Confidence Score", value=f"{max_prob*100:.2f}%")
                
                # Probability Distribution Breakdown
                st.markdown("##### Detailed Class Probabilities:")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = predictions[i] * 100
                    st.progress(int(round(prob)), text=f"{class_name}: {prob:.2f}%")
    else:
        st.info("👈 Please upload an image on the left panel to activate the classifier.")

# ==========================================
# 7. FOOTER & DISCLAIMER
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "<b>Disclaimer:</b> DermAI is developed as an academic Final Year Design Project (FYDP). "
    "It is intended solely for educational and research purposes and must not replace formal medical advice."
    "</div>", 
    unsafe_allow_html=True
)
