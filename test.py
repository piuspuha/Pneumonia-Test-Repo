import os
import random
import requests
import streamlit as st
from PIL import Image
import gdown

# Temporary folder to store downloaded images
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

# Streamlit app
def main():
    st.title("Google Drive Dynamic Image Viewer")

    # Input: Google Drive folder ID
    folder_id = st.text_input("Enter Google Drive Folder ID:")
    num_images = st.number_input("Number of images to display (fixed):", min_value=1, max_value=20, value=6)

    if st.button("Show Random Images"):
        if not folder_id:
            st.error("Please provide a valid Google Drive folder ID.")
            return

        # Fetch files from the folder
        with st.spinner("Fetching file list from Google Drive..."):
            files = get_files_from_drive_folder(folder_id)

        if not files:
            st.error("No image files found in the folder.")
            return

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

if __name__ == "__main__":
    main()
