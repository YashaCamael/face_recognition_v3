from flask import Blueprint, request, jsonify
from app.services.embedding import get_embedding
from app.utils.image_handler import load_image_from_base64, load_image_from_url, load_image_from_gcs

embedding_bp = Blueprint('embedding_bp', __name__)

@embedding_bp.route('/embedding', methods=['POST'])
def embedding_route():
    data = request.get_json()
    if not data or 'instances' not in data or not data['instances']:
        return jsonify({"predictions": [{"error": "Invalid payload format"}]}), 400

    instance = data['instances'][0]
    parameters = data.get('parameters', {})
    
    img_array = None
    try:
        if 'img_base64' in instance:
            img_array = load_image_from_base64(instance['img_base64'])
        elif 'img_link' in instance:
            img_array = load_image_from_url(instance['img_link'])
        elif 'img_gcs_uri' in instance:
            img_array = load_image_from_gcs(instance['img_gcs_uri'])
    except Exception as e:
        return jsonify({"predictions": [{"error": f"Failed to load image: {e}"}]}), 400

    if img_array is None:
        return jsonify({"predictions": [{"error": "No valid image data provided"}]}), 400

    result = get_embedding(img_array, parameters)
    return jsonify(result)