from flask import Blueprint, request, jsonify
from app.services.image_represent import represent_image
from app.utils.image_handler import load_image_from_base64, load_image_from_url, load_image_from_gcs

represent_bp = Blueprint('represent_bp', __name__)

@represent_bp.route('/represent', methods=['POST'])
def represent():
    """Receives image data and returns facial representation embeddings."""
    data = request.get_json()

    if not data or 'instances' not in data or not isinstance(data.get('instances'), list):
        return jsonify({'error': 'Invalid payload format'}), 400

    instances = data['instances']
    parameters = data.get('parameters', {})

    if not instances:
        return jsonify({'error': 'No instances provided'}), 400

    face_anti_spoofing = parameters.get('face_anti_spoofing', True)

    predictions = []

    for instance in instances:
        img_array = None
        try:
            if 'img_base64' in instance:
                img_array = load_image_from_base64(instance['img_base64'])
            elif 'img_link' in instance:
                img_array = load_image_from_url(instance['img_link'])
            elif 'img_gcs_uri' in instance:
                img_array = load_image_from_gcs(instance['img_gcs_uri'])

            if img_array is None:
                predictions.append({"error": "No valid image data provided in instance"})
                continue

            # Pass the renamed flag to the service function
            result = represent_image(img_array, parameters, face_anti_spoofing)

            if 'predictions' in result and isinstance(result['predictions'], list):
                predictions.extend(result['predictions'])
            else:
                predictions.append({"error": "Unexpected result format from service"})

        except Exception as e:
            predictions.append({"error": f"An unexpected error occurred: {e}"})

    if len(predictions) > 1:
        return jsonify({"predictions": [{"error": "Detected 2 or more faces"}]}), 400

    return jsonify({"predictions": predictions})