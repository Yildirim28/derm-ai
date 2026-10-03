import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. PAGE CONFIG & MODERN STYLING
# ==========================================
st.set_page_config(
    page_title="AI Skin Disease Classifier",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for modern look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: 700;
    }
    .sub-text {
        font-size: 1.1rem;
        color: #7f8c8d;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR TAB: PROJECT & TEAM DETAILS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/chatbot.png", width=80)
    st.title("Project Details")
    st.markdown("---")
    
    st.subheader("📌 About the Project")
    st.write(
        "This system is developed as a **Final Year Design Project (FYDP)** "
        "to provide preliminary skin disease assessments using deep learning."
    )
    
    st.markdown("---")
    st.subheader("⚙️ Model Specifications")
    st.markdown("""
    * **Architecture:** MobileNetV2 (Optimized)
    * **Input Resolution:** 224x224 px
    * **Target Classes:** 5 Categories
    * **Reliability Threshold:** 60% Confidence
    """)
    
    st.markdown("---")
    st.subheader("👥 Developer / Team")
    st.markdown("""
    * **Student ID:** 0112230663
    * **Institution:** United International University (UIU)
    * **Department:** CSE
    """)
    
    st.markdown("---")
    st.caption("© 2026 AI Healthcare Research Initiative")

# ==========================================
# 3. MAIN INTERFACE CONTENT
# ==========================================
st.markdown('<p class="main-header">🩺 AI-Based Skin Disease Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Upload a clear skin lesion image for a fast, AI-powered preliminary diagnosis.</p>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 4. LOAD MODEL (Cached)
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mobilenet_v2_optimized.h5')

with st.spinner("Initializing AI Engine... Please wait."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        st.info("💡 Tip: Ensure 'mobilenet_v2_optimized.h5' is placed in the root directory alongside 'app.py'.")
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
# 6. UPLOAD & PREDICTION WORKFLOW
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Skin Sample', use_column_width=True)

with col2:
    st.subheader("📊 Analysis Results")
    if uploaded_file is not None:
        if st.button("Run Classification", type="primary", use_container_width=True):
            with st.spinner("Analyzing dermatological features..."):
                processed_img = process_image(image)
                predictions = model.predict(processed_img)[0]
                
                max_prob = np.max(predictions)
                predicted_class = CLASS_NAMES[np.argmax(predictions)]
                
                st.markdown("---")
                
                # Uncertainty Handling (60% Threshold)
                if max_prob < 0.60:
                    st.warning("⚠️ **Uncertain Prediction Detected**")
                    st.write(f"Model confidence is **{max_prob*100:.2f}%** (Below the 60% threshold).")
                    st.info("💡 **Recommendation:** Please consult a certified dermatologist for verification.")
                else:
                    st.success(f"### Predicted Condition: **{predicted_class}**")
                    st.metric(label="Confidence Score", value=f"{max_prob*100:.2f}%")
                
                # Probability Distribution Breakdown
                st.write("#### Confidence Breakdown:")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = predictions[i] * 100
                    st.progress(int(round(prob)), text=f"{class_name}: {prob:.2f}%")
    else:
        st.info("👈 Please upload an image on the left panel to begin analysis.")

# ==========================================
# 7. FOOTER & DISCLAIMER
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85rem;'>"
    "Disclaimer: Developed as a Final Year Academic Project. Not intended for formal medical diagnosis."
    "</div>", 
    unsafe_allow_html=True
)
