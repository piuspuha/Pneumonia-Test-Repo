# Use the official Streamlit base image
FROM python:3.12-slim

# Install necessary dependencies
RUN pip install --upgrade pip
RUN pip install streamlit tensorflow==2.12.0 requests Pillow numpy

# Set the working directory
WORKDIR /app

# Copy the contents of the current directory to the Docker container
COPY . /app

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app_1.py"]

