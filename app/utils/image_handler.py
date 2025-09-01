import numpy as np
from PIL import Image
import base64
import requests
from flask import current_app
from google.cloud import storage
import io

def bytes_to_numpy_array(image_bytes: bytes) -> np.ndarray:
    """Decodes a byte string into a NumPy array using Pillow-SIMD."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Could not decode image from bytes: {e}")

def load_image_from_base64(base64_str: str) -> np.ndarray:
    """Decodes a Base64 string into a NumPy array."""
    try:
        image_bytes = base64.b64decode(base64_str)
        return bytes_to_numpy_array(image_bytes)
    except Exception as e:
        raise ValueError(f"Invalid Base64 string: {e}")

def load_image_from_url(url: str) -> np.ndarray:
    """Downloads an image from a URL into a NumPy array."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return bytes_to_numpy_array(response.content)

def load_image_from_gcs(gcs_uri: str, storage_client=None) -> np.ndarray:
    """
    Downloads an image from GCS into a NumPy array.
    If no storage_client is provided, it uses the one from the Flask app context.
    """
    if storage_client is None:
        storage_client = current_app.config['STORAGE_CLIENT']
    
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with 'gs://'")
    
    gcs_uri_path = gcs_uri[5:]
    bucket_name, gcs_image_path = gcs_uri_path.split('/', 1)
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_image_path)
    
    image_bytes = blob.download_as_bytes()
    
    return bytes_to_numpy_array(image_bytes)
