from deepface import DeepFace
from google.cloud.trace import Tracer

tracer = Tracer()

def get_embedding(img_array, parameters):
    """Generates a facial embedding for a given image array."""
    try:
        with tracer.span(name="deepface.represent"):
            embedding_obj = DeepFace.represent(
                img_path=img_array,
                model_name=parameters.get('model_name', 'Facenet512'),
                detector_backend=parameters.get('detector_backend', 'retinaface'),
                enforce_detection=parameters.get('enforce_detection', True),
                align=parameters.get('align', True),
                normalization=parameters.get('normalization', 'base')
            )
        
        if embedding_obj:
            result = {
                "embedding": embedding_obj[0].get("embedding", []),
                "facial_area": embedding_obj[0].get("facial_area", {}),
                "face_confidence": embedding_obj[0].get("face_confidence", 0.0)
            }
            return {"predictions": [result]}
        return {"predictions": []}
    except Exception as e:
        return {"predictions": [{"error": str(e)}]}