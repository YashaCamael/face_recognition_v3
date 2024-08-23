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
