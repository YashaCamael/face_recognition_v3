from deepface import DeepFace
from opentelemetry import trace
import tensorflow as tf
import logging

# Get a tracer for this specific module
tracer = trace.get_tracer(__name__)

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    logging.info(f"{len(physical_devices)} GPU(s) available.")
else:
    logging.warning("No GPU available, using CPU.")

def get_embedding(img_array, parameters):
    """Generates a facial embedding for a given image array."""
    try:
        # Create a custom span to measure this specific operation
        with tracer.start_as_current_span("deepface.represent"):
            embedding_obj = DeepFace.represent(
                img_path=img_array,
                model_name=parameters.get('model_name', 'Facenet512'),
                detector_backend=parameters.get('detector_backend', 'retinaface'),
                enforce_detection=parameters.get('enforce_detection', True),
                align=parameters.get('align', True),
                normalization=parameters.get('normalization', 'base')
            )

        if not embedding_obj:
            return {"predictions": [{"error": "Face could not be detected in the image."}]}

        if len(embedding_obj) > 1:
            return {"predictions": [{"error": "Detected 2 or more faces in the image."}]}

        result = {
            "embedding": embedding_obj[0].get("embedding", []),
            "facial_area": embedding_obj[0].get("facial_area", {}),
            "face_confidence": embedding_obj[0].get("face_confidence", 0.0)
        }
        return {"predictions": [result]}

    except ValueError as e:
        error_message = str(e)

        if 'Face could not be detected' in error_message:
            error_message = "Face could not be detected in the image. Please confirm that the picture is a face photo or set enforce_detection to False."

        else:
            error_message = "An error occurred while processing the image."

        # Include error in predictions array
        return {"predictions": [{"error": error_message, "embedding": [], "facial_area": {}, "face_confidence": 0.0}]}

    except Exception as e:
        # Include error in predictions array
        return {"predictions": [{"error": "An unexpected error occurred: " + str(e), "embedding": [], "facial_area": {}, "face_confidence": 0.0}]}
