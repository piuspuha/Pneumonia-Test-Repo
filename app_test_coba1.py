import streamlit as st
import tensorflow
from PIL import Image
import numpy as np
import os
import requests
import gdown
import random

#Temporary Downloaded Files Folder
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

key = 'AIzaSyDDwXywuCrDEQYTKwEcwVTkFcVmVpWpcZY'

# Function to get files from a public Google Drive folder
def get_files_from_drive_folder(folder_id):
    url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&key={key}&fields=files(id,name,mimeType)"
    response = requests.get(url)
    if response.status_code == 200:
        files = response.json().get("files", [])
        return [(file["id"], file["name"]) for file in files if "image" in file["mimeType"]]
    else:
        st.error("Failed to fetch files. Check your folder ID and API key.")
        return []

# Function to download an image using gdown
def download_image(file_id, file_name):
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join(TEMP_DIR, file_name)
    gdown.download(url, output_path, quiet=True)
    return output_path



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

    st.title("Google Drive Dynamic Image Viewer")

    # Input: Google Drive folder ID
    folder_select = st.sidebar.selectbox("Jenis X-Ray", ["Normal", "Penumonia"])

    if folder_select == "Normal":
        folder_id = "1DAyMmkAiPm9zVGy0K0_BnAeQxDNbl76X"
    elif folder_select == "Penumonia":
        folder_id = "1ESKX8q_7uMSf65yx_Q3SyopPbwZrmZy3"

    # folder_id = st.text_input("Enter Google Drive Folder ID:")
    num_images = st.number_input("Number of images to display (fixed):", min_value=1, max_value=20, value=6)

    if st.button("Show Random Images"):
        if not folder_id:
            st.error("Please provide a valid Google Drive folder ID.")

        # Fetch files from the folder
        with st.spinner("Fetching file list from Google Drive..."):
            files = get_files_from_drive_folder(folder_id)

        if not files:
            st.error("No image files found in the folder.")

        # Select random files
        selected_files = random.sample(files, min(num_images, len(files)))

        # Download and display images
        with st.spinner("Fetching files..."):
            downloaded_images = []
            for file_id, file_name in selected_files:
                try:
                    image_path = download_image(file_id, file_name)
                    downloaded_images.append((image_path, file_name))
                except Exception as e:
                    st.error(f"Error downloading {file_name}: {e}")

        # Display images in a grid
        cols = st.columns(3)
        for idx, (image_path, file_name) in enumerate(downloaded_images):
            col = cols[idx % 3]
            with col:
                image = Image.open(image_path)
                st.image(image, caption=file_name, use_column_width=True)


    # st.header("Visualisasi X-ray")
    
    # # User inputs for visualization
    # category = st.selectbox("Kategori", ["Pneumonia", "Normal"])
    # num_images = st.text_input("Jumlah gambar yang ingin ditampilkan:", value="9")
    
    # try:
    #     num_images = int(num_images)
    #     image_paths = load_dataset_images(category, num_images)
    #     st.subheader(f"Menampilkan {num_images} gambar kategori: {category}")
        
    #     # Display images in a grid layout (3 images per row)
    #     cols = st.columns(3)
    #     for idx, img_path in enumerate(image_paths):
    #         with cols[idx % 3]:
    #             image = Image.open(img_path)
    #             st.image(image, caption=os.path.basename(img_path), use_container_width=True)  # Changed parameter
    # except ValueError:
    #     st.error("Masukkan angka valid untuk jumlah gambar!")
