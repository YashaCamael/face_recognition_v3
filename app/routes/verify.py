from flask import Blueprint, request, jsonify, current_app
from app.services.face_verification import verify_faces
from app.utils.image_handler import save_image_from_base64, save_image_from_url
import uuid
import os

verify_bp = Blueprint('verify_bp', __name__)

@verify_bp.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()

    # Check if 'instances' exists and is a list
    if 'instances' not in data or not isinstance(data.get('instances'), list):
        return jsonify({'error': 'Invalid payload format, "instances" must be a list of dictionaries'}), 400

    instances = data['instances']
    parameters = data.get('parameters', {})
    
    if not instances:
        return jsonify({'error': 'No instances provided'}), 400

    results = []

    for instance in instances:
        # Generate unique filenames for the images
        unique_filename1 = f"{uuid.uuid4()}.jpg"
        unique_filename2 = f"{uuid.uuid4()}.jpg"

        img1_path = img2_path = None

        try:
            # Handle img1
            if 'img1_base64' in instance:
                img1_path = save_image_from_base64(instance['img1_base64'], unique_filename1)
            elif 'img1_link' in instance:
                img1_path = save_image_from_url(instance['img1_link'], unique_filename1)

            # Handle img2
            if 'img2_base64' in instance:
                img2_path = save_image_from_base64(instance['img2_base64'], unique_filename2)
            elif 'img2_link' in instance:
                img2_path = save_image_from_url(instance['img2_link'], unique_filename2)

            if not img1_path or not img2_path:
                return jsonify({'error': 'Please provide valid image data for both images'}), 400

            # Call your verification function
            result = verify_faces(img1_path, img2_path, parameters)
            results.append(result['predictions'][0])  # Extract the prediction result

        finally:
            # Ensure that any created files are cleaned up
            if img1_path and os.path.exists(img1_path):
                os.remove(img1_path)
            if img2_path and os.path.exists(img2_path):
                os.remove(img2_path)

    return jsonify({"predictions": results})
