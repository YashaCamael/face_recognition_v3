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

def get_image_from_instances(instance):
    if 'img_base64' in instance:
        return save_image_from_base64(instance['img_base64'], 'represent.jpg')
    elif 'img_link' in instance:
        return save_image_from_url(instance['img_link'], 'represent.jpg')
    return None

@app.route('/')
def home():
    return "Welcome to the Face Verification API!"

@app.route('/liveness', methods=['GET'])
def liveness():
    return jsonify({"status": "live"}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/represent', methods=['POST'])
def represent():
    data = request.get_json()
    
    if data is None or 'instances' not in data or not isinstance(data['instances'], list):
        return jsonify({'predictions': {'message': 'Invalid payload format, missing "instances" key or it is not a list'}}), 400

    if len(data['instances']) == 0:
        return jsonify({'predictions': {'message': 'Instances array is empty'}}), 400

    predictions = []
    for instance in data['instances']:
        img_path = get_image_from_instances(instance)
        if img_path is None:
            predictions.append({'message': 'Invalid image data provided'})
            continue

        parameters = data.get('parameters', {})
        model_name = parameters.get('model_name', 'Facenet512')
        detector_backend = parameters.get('detector_backend', 'mtcnn')
        enforce_detection = parameters.get('enforce_detection', True)
        align = parameters.get('align', True)

        try:
            representation = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align
            )
            predictions.append(representation)
        finally:
            os.remove(img_path)

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
