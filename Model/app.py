import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🌽 Maize Disease Detector")
model = YOLO("maize_disease_model_v2_final.pt")

uploaded_file = st.file_uploader("Upload a maize leaf...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    results = model(img) # Run inference
    
    # Plot results
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Detected Results", use_container_width=True)