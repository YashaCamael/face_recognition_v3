from flask import Flask, request, jsonify
import requests
from deepface import DeepFace
import os
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def save_image_from_base64(base64_str, filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img.save(img_path)
    return img_path

def save_image_from_url(url, filename):
    response = requests.get(url)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if response.status_code == 200:
        with open(img_path, 'wb') as f:
            f.write(response.content)
    return img_path

@app.route('/')
def home():
    return "Welcome to the Face Verification API!"

@app.route('/liveness', methods=['GET'])
def liveness():
    return jsonify({"status": "live"}), 200

@app.route('/health', methods=['GET'])
def health():
    # Implement any necessary checks to determine the health of your application
    return jsonify({"status": "healthy"}), 200

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    
    if 'instances' not in data or not isinstance(data.get('instances', {}), dict):
        return jsonify({'error': 'Invalid payload format, missing "instances" key or it is not a dictionary'}), 400

    instances = data['instances']
    parameters = data.get('parameters', {})

    # Determine source of images
    img1_path = img2_path = None
    if 'img1_base64' in instances:
        img1_path = save_image_from_base64(instances['img1_base64'], 'img1.jpg')
    elif 'img1_link' in instances:
        img1_path = save_image_from_url(instances['img1_link'], 'img1.jpg')
    
    if 'img2_base64' in instances:
        img2_path = save_image_from_base64(instances['img2_base64'], 'img2.jpg')
    elif 'img2_link' in instances:
        img2_path = save_image_from_url(instances['img2_link'], 'img2.jpg')

    if not img1_path or not img2_path:
        return jsonify({'error': 'Please provide valid image data for both images'}), 400

    result = DeepFace.verify(
        img1_path, 
        img2_path,
        model_name=parameters.get('model_name', 'VGG-Face'),
        detector_backend=parameters.get('detector_backend', 'opencv'),
        distance_metric=parameters.get('distance_metric', 'cosine'),
        enforce_detection=parameters.get('enforce_detection', True),
        align=parameters.get('align', True),
        normalization=parameters.get('normalization', 'base'),
        silent=parameters.get('silent', True),
        threshold=parameters.get('threshold', None)
    )

    # Cleanup image files
    os.remove(img1_path)
    os.remove(img2_path)

    response = {"predictions": [result]}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
