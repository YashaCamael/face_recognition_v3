from flask import Blueprint, request, jsonify
from app.services.face_verification import verify_faces
from app.utils.image_handler import load_image_from_base64, load_image_from_url

verify_bp = Blueprint('verify_bp', __name__)

@verify_bp.route('/verify', methods=['POST'])
def verify():
    """Receives two images and verifies if they are of the same person."""
    data = request.get_json()

    if not data or 'instances' not in data or not isinstance(data.get('instances'), list):
        return jsonify({'error': 'Invalid payload format'}), 400

    instances = data['instances']
    parameters = data.get('parameters', {})
    
    if not instances:
        return jsonify({'error': 'No instances provided'}), 400

    results = []
    for instance in instances:
        img1_array = None
        img2_array = None

        try:
            # Handle img1
            if 'img1_base64' in instance:
                img1_array = load_image_from_base64(instance['img1_base64'])
            elif 'img1_link' in instance:
                img1_array = load_image_from_url(instance['img1_link'])

            # Handle img2
            if 'img2_base64' in instance:
                img2_array = load_image_from_base64(instance['img2_base64'])
            elif 'img2_link' in instance:
                img2_array = load_image_from_url(instance['img2_link'])

            if img1_array is None or img2_array is None:
                results.append({"error": "Please provide valid image data for both images"})
                continue

            # Call the verification function with image arrays
            result = verify_faces(img1_array, img2_array, parameters)
            results.append(result['predictions'][0])

        except Exception as e:
            results.append({"error": f"An unexpected error occurred: {e}", "verified": False})

    return jsonify({"predictions": results})