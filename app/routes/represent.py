from flask import Blueprint, request, jsonify
from app.services.image_represent import represent_image
from app.utils.image_handler import save_image_from_base64, save_image_from_url
import uuid
import os

represent_bp = Blueprint('represent_bp', __name__)

@represent_bp.route('/represent', methods=['POST'])
def represent():
    data = request.get_json()

    # Check if 'instances' exists and is a list
    if 'instances' not in data or not isinstance(data.get('instances'), list):
        return jsonify({'error': 'Invalid payload format, "instances" must be a list of dictionaries'}), 400

    instances = data['instances']
    parameters = data.get('parameters', {})
    
    if not instances:
        return jsonify({'error': 'No instances provided'}), 400

    predictions = []

    for instance in instances:
        # Generate a unique filename for the image
        unique_filename = f"{uuid.uuid4()}.jpg"

        img_path = None

        try:
            # Handle image input
            if 'img_base64' in instance:
                img_path = save_image_from_base64(instance['img_base64'], unique_filename)
            elif 'img_link' in instance:
                img_path = save_image_from_url(instance['img_link'], unique_filename)

            if not img_path:
                predictions.append({"error": "Please provide valid image data for one or more images"})
                continue  # Skip to the next instance

            # Call the represent function
            result = represent_image(img_path, parameters)

            # Ensure the result is a dictionary with 'predictions' as a list
            if 'predictions' in result and isinstance(result['predictions'], list):
                # Append the content of 'predictions' list
                predictions.extend(result['predictions'])
            else:
                predictions.append({"error": "Unexpected result format from represent_image function"})

        except Exception as e:
            predictions.append({"error": "An unexpected error occurred: " + str(e)})

        finally:
            # Ensure that any created files are cleaned up
            if img_path and os.path.exists(img_path):
                os.remove(img_path)

    # Check if there are multiple faces detected
    if len(predictions) > 1:
        return jsonify({"predictions": [{"error": "Detected 2 or more photos KTP"}]}), 400

    return jsonify({"predictions": predictions})
