import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="Maize Disease Detector", page_icon="🌽", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stHeading { color: #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌽 Maize Disease Detector")
st.markdown("### Early detection for East African food security")

# Load your 3.0MB lightweight model
@st.cache_resource
def load_model():
    return YOLO("maize_disease_model_v2_final.pt")

model = load_model()

# Sidebar for information
with st.sidebar:
    st.header("Project Info")
    st.info("This model is optimized for edge deployment in East Africa, identifying MSV and NLB with 99.3% accuracy.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Corn_leaf.jpg/320px-Corn_leaf.jpg", caption="Healthy Maize")

# Main UI
uploaded_file = st.file_uploader("Upload a maize leaf photo (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Original Upload", use_container_width=True)
    
    with col2:
        with st.spinner('Analyzing symptoms...'):
            results = model(img)
            # YOLOv8 returns a list, we take the first result
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detection Results", use_container_width=True)

    # Display specific diagnosis
    st.divider()
    probs = results[0].probs  # Access classification probabilities
    if probs is not None:
        class_id = probs.top1
        label = results[0].names[class_id]
        conf = probs.top1conf.item()

        st.subheader(f"Diagnosis: **{label}**")
        st.metric(label="Confidence Level", value=f"{conf:.2%}")

        if label == "Maize_Streak_Virus":
            st.warning("**Note:** Look for tiny pale spots or signature yellow streaks along veins.")
        elif label == "Maize_Blight":
            st.warning("**Note:** Look for long, narrow 'cigar-shaped' tan or grey lesions.")
        else:
            st.success("The leaf appears healthy.")