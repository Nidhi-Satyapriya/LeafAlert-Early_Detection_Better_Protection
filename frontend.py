import streamlit as st
import requests
from PIL import Image

# FastAPI endpoint URL
API_URL = "http://localhost:8000/predict"  # Update if FastAPI is running on a different host or port

# Streamlit app setup
st.title("Potato Disease Classification")
st.write("Upload an image to classify potato diseases")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Send image to FastAPI server for prediction
    with st.spinner("Classifying..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Class: {result['class']}")
            st.write(f"Confidence: {result['confidence']:.2f}")
        else:
            st.error("Prediction failed. Please try again.")
#