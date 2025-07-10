from deepface import DeepFace

def verify_faces(img1_array, img2_array, parameters):
    """Verifies if two image arrays contain the same face."""
    try:
        result = DeepFace.verify(
            img1_path=img1_array, 
            img2_path=img2_array,
            model_name=parameters.get('model_name', 'Facenet512'),
            detector_backend=parameters.get('detector_backend', 'opencv'),
            distance_metric=parameters.get('distance_metric', 'euclidean_l2'),
            enforce_detection=parameters.get('enforce_detection', True),
            align=parameters.get('align', True),
            normalization=parameters.get('normalization', 'base')
        )

        simplified_result = {
            "distance": result.get("distance"),
            "verified": result.get("verified"),
            "model": result.get("model"),
            "similarity_metric": result.get("similarity_metric"),
            "detector_backend": result.get("detector_backend"),
            "threshold": result.get("threshold"),
            "time": result.get("time")
        }

        return {"predictions": [simplified_result]}
    
    except ValueError as e:
        error_message = str(e)
        if 'Face could not be detected' in error_message:
            error_message = "Face could not be detected in one or both images."
        elif 'Spoof detected' in error_message:
            error_message = "Spoof detected in one of the images."
        
        return {"predictions": [{"error": error_message, "verified": False}]}

    except Exception as e:
        return {"predictions": [{"error": f"An unexpected error occurred: {e}", "verified": False}]}