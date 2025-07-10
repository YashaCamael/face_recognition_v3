import logging
import concurrent.futures
from deepface import DeepFace

def process_represent_result(result):
    """Formats the raw DeepFace result into a simplified list of dictionaries."""
    simplified_result = []
    if isinstance(result, list):
        for res_item in result:
            if isinstance(res_item, dict):
                face_data = {
                    "embedding": res_item.get("embedding", []),
                    "facial_area": res_item.get("facial_area", {}),
                    "face_confidence": res_item.get("face_confidence", 0.0)
                }
                simplified_result.append(face_data)
            else:
                logging.warning(f"Unexpected item type in result list: {type(res_item)}")
    elif isinstance(result, dict):
        face_data = {
            "embedding": result.get("embedding", []),
            "facial_area": result.get("facial_area", {}),
            "face_confidence": result.get("face_confidence", 0.0)
        }
        simplified_result.append(face_data)
    return simplified_result

def task_run_antispoof(img_array):
    """Task to run anti-spoofing check."""
    try:
        result = DeepFace.extract_faces(
            img_path=img_array,
            detector_backend='retinaface',
            enforce_detection=True,
            anti_spoofing=True
        )
        if result:
            first_item = result[0]
            return first_item.get('is_real'), first_item.get('antispoof_score')
    except Exception as e:
        logging.info(f"Anti-spoofing task failed: {e}")
    return None, None

def task_run_represent(img_array, parameters):
    """Task to generate facial embeddings."""
    result = DeepFace.represent(
        img_array,
        model_name=parameters.get('model_name', 'Facenet512'),
        detector_backend=parameters.get('detector_backend', 'retinaface'),
        enforce_detection=parameters.get('enforce_detection', True),
        align=parameters.get('align', True),
        normalization=parameters.get('normalization', 'base')
    )
    return process_represent_result(result)

def represent_image(img_array, parameters, face_anti_spoofing: bool):
    """Generates facial embeddings, conditionally running anti-spoofing in parallel."""
    is_real = None
    antispoof_score = None
    predictions_list = []

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            future_antispoof = None
            if face_anti_spoofing:
                # Only submit the anti-spoofing task if enabled
                future_antispoof = executor.submit(task_run_antispoof, img_array)
            
            future_represent = executor.submit(task_run_represent, img_array, parameters)

            if future_antispoof:
                is_real, antispoof_score = future_antispoof.result()
            
            predictions_list = future_represent.result()

        if predictions_list:
            for prediction_item in predictions_list:
                prediction_item['is_real'] = is_real
                prediction_item['antispoof_score'] = antispoof_score
        
        return {"predictions": predictions_list}

    except ValueError as e:
        error_message_str = str(e)
        final_error_message = ""
        if 'Face could not be detected' in error_message_str:
            final_error_message = "Face could not be detected in the image."
        elif 'Spoof detected' in error_message_str:
            final_error_message = "Spoof detected in the given image."
        else:
            final_error_message = f"A ValueError occurred: {error_message_str}"
        return {"predictions": [{"error": final_error_message, "embedding": None, "facial_area": None, "face_confidence": None, "is_real": is_real, "antispoof_score": antispoof_score}]}

    except Exception as e:
        return {"predictions": [{"error": "An unexpected error occurred: " + str(e), "embedding": None, "facial_area": None, "face_confidence": None, "is_real": is_real, "antispoof_score": antispoof_score}]}