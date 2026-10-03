import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Skin Disease Classifier",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI-Based Skin Disease Classification System")
st.write("Upload a clear skin lesion image to get an AI-powered preliminary assessment using our optimized MobileNetV2 model.")

# ==========================================
# 2. LOAD MODEL (Cached for fast performance)
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mobilenet_v2_optimized.h5')

with st.spinner("Loading AI Model... Please wait."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        st.stop()

CLASS_NAMES = ['Acne', 'Eczema', 'Melanoma', 'Normal Skin', 'Psoriasis']

# ==========================================
# 3. IMAGE PREPROCESSING FUNCTION
# ==========================================
def process_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# ==========================================
# 4. USER INTERFACE & PREDICTION
# ==========================================
uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Skin Image', use_column_width=True)
    
    if st.button("Run Classification", type="primary"):
        with st.spinner("Analyzing image features..."):
            processed_img = process_image(image)
            predictions = model.predict(processed_img)[0]
            
            max_prob = np.max(predictions)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            
            st.divider()
            
            # ==========================================
            # 5. UNCERTAINTY HANDLING (60% Threshold)
            # ==========================================
            if max_prob < 0.60:
                st.warning("⚠️ **Uncertain Prediction Detected**")
                st.write(f"The model confidence is **{max_prob*100:.2f}%**, which is below our **60% reliability threshold**.")
                st.info("💡 **Recommendation:** As the model is uncertain, please consult a certified dermatologist for an accurate clinical diagnosis.")
            else:
                st.success(f"### Predicted Condition: **{predicted_class}**")
                st.metric(label="Confidence Score", value=f"{max_prob*100:.2f}%")
            
            # ==========================================
            # 6. PROBABILITY DISTRIBUTION BREAKDOWN
            # ==========================================
            st.write("#### Detailed Probability Distribution:")
            for i, class_name in enumerate(CLASS_NAMES):
                prob = predictions[i] * 100
                st.progress(int(round(prob)), text=f"{class_name}: {prob:.2f}%")

# ==========================================
# 7. FOOTER & DISCLAIMER
# ==========================================
st.markdown("---")
st.caption("Disclaimer: This tool is developed as a final-year academic project for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.")
