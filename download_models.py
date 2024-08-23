import os
import requests

def download_file(url, destination):
    """Download a file from a given URL and save it to the destination."""
    response = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {url} to {destination}")

# Set up paths
home = os.path.expanduser("~")
destination_path_facenet = os.path.join(home, ".deepface/weights/facenet512_weights.h5")

# Download the files from public URLs
download_file("https://storage.googleapis.com/model_face_recognition/facenet512_weights.h5", destination_path_facenet)