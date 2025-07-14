from deepface import DeepFace
from google.cloud.trace import Tracer

tracer = Tracer()

def get_antispoof_data(img_array):
    """Performs an anti-spoofing check on a given image array."""
    try:
        with tracer.span(name="deepface.extract_faces.antispoof"):
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