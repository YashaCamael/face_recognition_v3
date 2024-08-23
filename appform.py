from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def save_image_from_url(url, filename):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img.save(img_path)
    return img_path

@app.route('/')
def home():
    return "homepage"

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    
    if 'instances' not in data or 'parameters' not in data:
        return jsonify({'error': 'Invalid payload format, missing "instances" or "parameters" key'}), 400

    instances = data['instances']
    parameters = data['parameters']

    # if not isinstance(instances, list) or len(instances) == 0:
    #     return jsonify({'error': 'The "instances" key must be a non-empty list'}), 400

    # instance = instances[0]  # Assuming only one instance for simplicity
    img1_path = None
    img2_path = None

    if 'img1_url' in instances:
        img1_url = instances['img1_url']
        img1_path = save_image_from_url(img1_url, 'img1.jpg')

    if 'img2_url' in instances:
        img2_url = instances['img2_url']
        img2_path = save_image_from_url(img2_url, 'img2.jpg')

    if not img1_path or not img2_path:
        return jsonify({'error': 'Please provide both image URLs'}), 400

    # Extract additional parameters from 'parameters' key
    model_name = parameters.get('model_name', 'VGG-Face')
    detector_backend = parameters.get('detector_backend', 'opencv')
    distance_metric = parameters.get('distance_metric', 'cosine')
    enforce_detection = parameters.get('enforce_detection', True)
    align = parameters.get('align', True)
    normalization = parameters.get('normalization', 'base')
    silent = parameters.get('silent', True)
    threshold = parameters.get('threshold', None)

    # Perform verification
    result = DeepFace.verify(
        img1_path, 
        img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align,
        normalization=normalization,
        silent=silent,
        threshold=threshold
    )

    os.remove(img1_path)
    os.remove(img2_path)

    # Wrapping the result in the expected Vertex AI format
    response = {
        "predictions": [result]
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
