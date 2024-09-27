import os
from deepface import DeepFace
from flask import jsonify

def process_represent_result(result):
    simplified_result = []

    # Check if result is a list of detected faces
    if isinstance(result, list):
        for res in result:
            # Process each detected face
            face_data = {
                "embedding": res.get("embedding", []),
                "facial_area": res.get("facial_area", {}),
                "face_confidence": res.get("face_confidence", 0.0)
            }
            simplified_result.append(face_data)
    else:
        # Handle case where result is not a list, e.g., an empty response or a single face
        simplified_result.append({
            "embedding": result.get("embedding", []),
            "facial_area": result.get("facial_area", {}),
            "face_confidence": result.get("face_confidence", 0.0)
        })

    return simplified_result


def represent_image(img_path, parameters):
    try:
        # Fetch anti_spoofing from parameters and convert it to boolean
        anti_spoofing_str = parameters.get('anti_spoofing', 'false')  # Default to 'false' if not provided
        anti_spoofing = anti_spoofing_str.lower() == 'true'  # Convert to boolean

        # Extract features from the image using DeepFace
        result = DeepFace.represent(
            img_path,
            model_name=parameters.get('model_name', 'Facenet512'),
            detector_backend=parameters.get('detector_backend', 'retinaface'),
            enforce_detection=parameters.get('enforce_detection', True),
            align=parameters.get('align', True),
            normalization=parameters.get('normalization', 'base'),
            anti_spoofing=anti_spoofing  # Use the converted boolean value
        )
        
        # Process the result
        return {"predictions": process_represent_result(result)}
    
    except ValueError as e:
        error_message = str(e)
        
        if 'Face could not be detected' in error_message:
            error_message = "Face could not be detected in the image. Please confirm that the picture is a face photo or set enforce_detection to False."
        
        # Add spoof detection error handling
        elif 'Spoof detected' in error_message:
            error_message = "Spoof detected in the given image. Please ensure that the image is not altered or manipulated."

        else:
            error_message = "An error occurred while processing the image."

        # Include error in predictions array
        return {"predictions": [{"error": error_message, "embedding": [], "facial_area": {}, "face_confidence": 0.0}]}

    except Exception as e:
        # Include error in predictions array
        return {"predictions": [{"error": "An unexpected error occurred: " + str(e), "embedding": [], "facial_area": {}, "face_confidence": 0.0}]}
    
    finally:
        # Cleanup image files after processing
        if os.path.exists(img_path):
            os.remove(img_path)
