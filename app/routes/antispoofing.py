from flask import Blueprint, request, jsonify
from app.services.antispoofing import get_antispoof_data
from app.utils.image_handler import load_image_from_base64, load_image_from_url, load_image_from_gcs
from app.utils.logger import log_api_interaction

antispoof_bp = Blueprint('antispoof_bp', __name__)

@antispoof_bp.route('/antispoof', methods=['POST'])
def antispoof_route():
    data = request.get_json()
    if not data or 'instances' not in data or not data['instances']:
        response = {"predictions": [{"error": "Invalid payload format"}]}
        log_api_interaction('antispoofing/antispoof', data, response, 400)
        return jsonify(response), 400

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
        error_response = {"predictions": [{"error": f"Failed to load image: {e}"}]}
        log_api_interaction('antispoofing/antispoof', data, error_response, 400)
        return jsonify(error_response), 400

    if img_array is None:
        error_response = {"predictions": [{"error": "No valid image data provided"}]}
        log_api_interaction('antispoofing/antispoof', data, error_response, 400)
        return jsonify(error_response), 400
        
    result = get_antispoof_data(img_array)
    log_api_interaction('antispoofing/antispoof', data, result, 200)
    return jsonify(result)