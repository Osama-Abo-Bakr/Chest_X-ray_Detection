# --------------------------------------------------------------------------------------------------------------------------
## ====================================== Streamlit Code =====================================================================

import streamlit as st
import requests as req
import tempfile
from PIL import Image

# Page Title And Configuration
st.set_page_config(page_title='Chest X-ray Detection', page_icon='🫁', layout='wide')
st.title('Chest X-ray Detection 🫁')

# Sidebar Section
st.sidebar.subheader("Upload Image")
file_uploader = st.sidebar.file_uploader('Upload the Image', type=['jpg', 'png', 'jpeg'])

if file_uploader:
    try:
        with st.spinner("Waiting for the prediction..."):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(file_uploader.read())
                temp_path = temp_file.name

            # Open the image to display in the app
            image = Image.open(temp_path)
            
            # Send the image file to the FastAPI backend
            with open(temp_path, "rb") as img_file:
                response = req.post(
                    "http://127.0.0.1:8000/Chest_Xray_Detection",  # Update this URL if deployed
                    files={"file": img_file}  # Use "file" to match the backend parameter name
                )
            
            if response.status_code == 200:
                prediction = response.json()
                st.info(
                    f"Prediction: {prediction['Prediction']} | "
                    f"Probability: {prediction['Probability']}%", 
                    icon='🫁'
                )
                st.image(image, caption='Uploaded Image', width=550)
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")