from flask import Blueprint, request, jsonify
from app.services.antispoofing_service import get_antispoof_data
from app.utils.image_handler import load_image_from_base64, load_image_from_url, load_image_from_gcs

antispoof_bp = Blueprint('antispoof_bp', __name__)

@antispoof_bp.route('/antispoof', methods=['POST'])
def antispoof_route():
    data = request.get_json()
    if not data or 'instances' not in data or not data['instances']:
        return jsonify({"predictions": [{"error": "Invalid payload format"}]}), 400

    instance = data['instances'][0]
    
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
        
    result = get_antispoof_data(img_array)
    return jsonify(result)