from flask import Blueprint, request, jsonify
from app.services.embedding import get_embedding
from app.utils.image_handler import load_image_from_base64, load_image_from_url, load_image_from_gcs
from app.utils.image_enhancer import preprocess_low_light, is_low_light
from app.utils.logger import log_api_interaction

embedding_bp = Blueprint('embedding_bp', __name__)

@embedding_bp.route('/represent', methods=['POST'])
def embedding_route():
    # --- Get the request data ---
    data = request.get_json()
    if not data or 'instances' not in data or not isinstance(data['instances'], list):
        response = {"predictions": [{"error": "Invalid payload format"}]}
        log_api_interaction('embedding/represent', data, response, 400)
        return jsonify(response), status_code, 400

    # --- Prepare to collect results for all instances ---
    all_predictions = []
    parameters = data.get('parameters', {})

    # --- Get low-light mode settings ---
    low_light_mode = parameters.get("low_light_mode", "auto")
    brightness_threshold = parameters.get("brightness_threshold", 70) # Make threshold configurable
    
    # --- Loop through each instance in the batch ---
    for instance in data['instances']:
        img_array = None
        error_message = None

        # --- Load image for the current instance ---
        try:
            if 'img_base64' in instance:
                img_array = load_image_from_base64(instance['img_base64'])
            elif 'img_link' in instance:
                img_array = load_image_from_url(instance['img_link'])
            elif 'img_gcs_uri' in instance:
                img_array = load_image_from_gcs(instance['img_gcs_uri'])
            else:
                error_message = "No valid image key (img_base64, img_link, img_gcs_uri) provided in instance"
        
        except Exception as e:
            error_message = f"Failed to load image: {e}"

        # --- APPLY PREPROCESSING ---
        if img_array is not None and not error_message:
            
            # Decide if we should apply the filter
            apply_preprocessing = False
            if low_light_mode == "force_on":
                apply_preprocessing = True
            elif low_light_mode == "auto":
                # Only apply if the automatic check says it's low light
                if is_low_light(img_array, threshold=brightness_threshold):
                    apply_preprocessing = True
            # Note: if mode is "force_off" or anything else, apply_preprocessing stays False
            
            # Run the filter if we decided to
            if apply_preprocessing:
                try:
                    img_array = preprocess_low_light(img_array)
                except Exception as e:
                    error_message = f"Failed during low-light preprocessing: {e}"

        # --- Get embedding or append the error for this instance ---
        if error_message:
            # If there was an error loading the image, add an error prediction
            all_predictions.append({"error": error_message})
        elif img_array is not None:
            # If image loaded successfully, get the embedding
            result = get_embedding(img_array, parameters)
            # The get_embedding function returns {"predictions": [...]}, so we extend our list
            all_predictions.extend(result.get("predictions", []))
        else:
            # Fallback for an unknown image loading issue
            all_predictions.append({"error": "No valid image data provided"})

    # --- Return the collected list of all predictions ---
    response = {"predictions": all_predictions}
    from app.utils.logger import check_has_error
    status_code = 400 if check_has_error(response) else 200
    log_api_interaction('embedding/represent', data, response, status_code)
    return jsonify(response), status_code