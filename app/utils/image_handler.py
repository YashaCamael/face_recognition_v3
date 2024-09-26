import os
import base64
from PIL import Image
from io import BytesIO
from flask import current_app
import requests

# Dictionary that maps image signatures to MIME types
image_signatures = {
    "iVBORw0KGgo": "image/png",  # PNG
    "/9j/": "image/jpeg"         # JPEG
}

# Function to detect MIME type based on the Base64 signature
def detect_mime_type(b64_str):
    for signature, mime_type in image_signatures.items():
        if b64_str.startswith(signature):
            return mime_type
    return None

# Function to save image from Base64 string with appropriate file extension
def save_image_from_base64(base64_str, filename_base):
    # Detect the MIME type from the Base64 string
    mime_type = detect_mime_type(base64_str[:20])  # Check only the first 20 chars
    if mime_type is None:
        raise ValueError("Unsupported image format")

    # Define the file extension based on the MIME type
    file_extension = ".png" if mime_type == "image/png" else ".jpg"
    
    # Combine the base filename and the file extension
    filename = f"{filename_base}{file_extension}"
    
    # Define the path to save the image
    img_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    # Decode the Base64 string and save the image
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img.save(img_path)
    
    return img_path

# Function to save image from URL
def save_image_from_url(url, filename):
    response = requests.get(url)
    img_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    if response.status_code == 200:
        with open(img_path, 'wb') as f:
            f.write(response.content)
    return img_path

# Function to download image from GCS
def download_image_from_gcs(gcs_uri):
    """
    Downloads an image from Google Cloud Storage using the initialized storage client in Flask config.
    
    :param gcs_uri: The GCS URI in the format 'gs://bucket_name/path/to/image.jpg'
    :return: Path to the downloaded image
    """
    # Extract the bucket name and object path from the GCS URI
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with 'gs://'")
    
    # Remove the 'gs://' prefix
    gcs_uri = gcs_uri[5:]
    
    # Split the URI into bucket and file path
    bucket_name, gcs_image_path = gcs_uri.split('/', 1)
    
    # Access the Google Cloud Storage client from Flask app config
    storage_client = current_app.config['STORAGE_CLIENT']
    
    # Access the bucket and the blob (file)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_image_path)
    
    # Define the local filename and path where the image will be saved
    filename = os.path.basename(gcs_image_path)  # Use the image name from GCS path
    local_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    # Download the image from GCS
    blob.download_to_filename(local_image_path)
    
    return local_image_path
