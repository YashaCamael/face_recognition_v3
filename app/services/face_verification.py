import os
from deepface import DeepFace
from flask import jsonify

def verify_faces(img1_path, img2_path, parameters):
    try:
        result = DeepFace.verify(
            img1_path, 
            img2_path,
            model_name=parameters.get('model_name', 'Facenet512'),
            detector_backend=parameters.get('detector_backend', 'opencv'),
            distance_metric=parameters.get('distance_metric', 'euclidean_l2'),
            enforce_detection=parameters.get('enforce_detection', True),
            align=parameters.get('align', True),
            normalization=parameters.get('normalization', 'base'),
            silent=parameters.get('silent', True),
            threshold=parameters.get('threshold', None)
        )

        # Extract relevant information for Vertex AI response
        simplified_result = {
            "distance": result.get("distance"),
            "verified": result.get("verified"),
            "model": result.get("model"),
            "similarity_metric": result.get("similarity_metric"),
            "detector_backend": result.get("detector_backend"),
            "threshold": result.get("threshold"),
            "time": result.get("time")
        }

        # Return the formatted prediction response
        return {"predictions": [simplified_result]}
    
    except ValueError as e:
        error_message = str(e)

        if 'Face could not be detected' in error_message:
            if img1_path in error_message:
                error_message = "Face could not be detected in img1. Please confirm that the picture is a face photo or set enforce_detection to False."
            elif img2_path in error_message:
                error_message = "Face could not be detected in img2. Please confirm that the picture is a face photo or set enforce_detection to False."
            else:
                error_message = "Face could not be detected in one of the images. Please confirm that the picture is a face photo or set enforce_detection to False."
        
        elif 'Exception while processing img1_path' in error_message:
            error_message = "An exception occurred while processing img1. Please check the image file or try again later."

        elif 'Exception while processing img2_path' in error_message:
            error_message = "An exception occurred while processing img2. Please check the image file or try again later."

        # Handle spoof detection error
        elif 'Spoof detected' in error_message:
            error_message = "Spoof detected in the given image. Please ensure that the image is not altered or manipulated."

        # Include error and verification status in predictions array
        return {"predictions": [{"error": error_message, "verified": False}]}

    except Exception as e:
        # Include error and verification status in predictions array
        return {"predictions": [{"error": "An unexpected error occurred: " + str(e), "verified": False}]}
    
    finally:
        # Cleanup image files after processing (successful or not)
        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)
