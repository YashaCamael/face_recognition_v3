import os
import base64
from PIL import Image
from io import BytesIO
from flask import current_app
import requests

def save_image_from_base64(base64_str, filename):
    img_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img.save(img_path)
    return img_path

def save_image_from_url(url, filename):
    response = requests.get(url)
    img_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    if response.status_code == 200:
        with open(img_path, 'wb') as f:
            f.write(response.content)
    return img_path

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
