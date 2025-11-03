from deepface import DeepFace
import tensorflow as tf
import logging

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    logging.info(f"{len(physical_devices)} GPU(s) available.")
else:
    logging.warning("No GPU available, using CPU.")

def get_antispoof_data(img_array):
    """Performs an anti-spoofing check on a given image array."""
    try:
        result = DeepFace.extract_faces(
            img_path=img_array,
            detector_backend='retinaface',
            enforce_detection=True,
            anti_spoofing=True
        )
        
        if result:
            first_face = result[0]
            spoof_data = {
                "is_real": first_face.get('is_real'),
                "antispoof_score": first_face.get('antispoof_score')
            }
            return {"predictions": [spoof_data]}
        return {"predictions": []}
    except Exception as e:
        return {"predictions": [{"error": str(e)}]}
