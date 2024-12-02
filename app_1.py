import streamlit as st
import tensorflow
from PIL import Image
import numpy as np
import os
import requests

# Google Drive model download link
MODEL_URL = "https://drive.google.com/uc?id=15qmsFIXmRgxMTpZ96TuidUWk_64MkN0C"

# Function to download the model from Google Drive
def download_model(model_url, model_path):
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)

# Download model if it does not exist
model_path = 'model.h5'  # Specify your desired model file path
if not os.path.exists(model_path):
    download_model(MODEL_URL, model_path)
    print(f"Model downloaded to {model_path}")

# Load the trained model
model = tensorflow.keras.models.load_model(model_path)

# Function to preprocess the input image
def preprocess_image(image):
    # Ensure the image has 3 color channels (RGB)
    image = image.convert("RGB")  # Converts grayscale images to RGB
    image = image.resize((150, 150))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Load sample dataset for visualization
def load_dataset_images(category, num_images):
    # Base folder for the dataset
    base_folder = r'pneumonia-detection\xray'
    folder = os.path.join(base_folder, 'PNEUMONIA') if category == 'Pneumonia' else os.path.join(base_folder, 'NORMAL')
    
    # Get the list of image paths
    images = os.listdir(folder)[:num_images]
    image_paths = [os.path.join(folder, img) for img in images]
    return image_paths

# App layout
st.title("X-ray Pneumonia Detection")

# Menu as a dropdown
menu = st.sidebar.selectbox("Menu", ["Prediksi", "Visualisasi"])

if menu == "Prediksi":
    st.header("Pneumonia Prediction")
    
    # File uploader for multiple image inputs
    uploaded_files = st.file_uploader("Upload X-ray Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    # Process and predict for each image
    if uploaded_files:
        cols = st.columns(3)  # Create three columns for displaying the images in a row
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            
            # Preprocess the image before making predictions
            processed_image = preprocess_image(image)
            
            # Get the prediction probabilities
            predictions = model.predict(processed_image)
            
            # The output is a probability distribution; get the predicted class and probability
            predicted_class = 'Pneumonia' if predictions[0][0] < 0.5 else 'Normal'
            probability = predictions[0][0] if predictions[0][0] < 0.5 else 1 - predictions[0][0]
            
            # Display the image and its prediction probability
            with cols[i % 3]:  # Use modulo to cycle through columns
                st.image(image, caption=f"{uploaded_file.name}", use_container_width=True)
                # Centered and custom font size for prediction and probability
                st.markdown(
                    f"""
                    <h3 style="text-align: center; font-size: 20px;">Prediction: {predicted_class}</h3>
                    <h3 style="text-align: center; font-size: 15px;">Pneumonia Probability: {probability:.4f}</h3>
                    """, 
                    unsafe_allow_html=True
                )

elif menu == "Visualisasi":
    st.header("Visualisasi X-ray")
    
    # User inputs for visualization
    category = st.selectbox("Kategori", ["Pneumonia", "Normal"])
    num_images = st.text_input("Jumlah gambar yang ingin ditampilkan:", value="9")
    
    try:
        num_images = int(num_images)
        image_paths = load_dataset_images(category, num_images)
        st.subheader(f"Menampilkan {num_images} gambar kategori: {category}")
        
        # Display images in a grid layout (3 images per row)
        cols = st.columns(3)
        for idx, img_path in enumerate(image_paths):
            with cols[idx % 3]:
                image = Image.open(img_path)
                st.image(image, caption=os.path.basename(img_path), use_container_width=True)  # Changed parameter
    except ValueError:
        st.error("Masukkan angka valid untuk jumlah gambar!")
