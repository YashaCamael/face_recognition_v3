from flask import Flask
import logging
import sys
import os
from app.services.face_verification import verify_faces
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

def create_app():
    # Configure logging to use stdout and set the log level to INFO
    logging.basicConfig(
        stream=sys.stdout,  # Send logs to stdout
        level=logging.INFO,  # Set the default log level to INFO
        # format='[%(levelname)s] %(name)s: %(message)s'
    )

    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads/'

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    from app.routes.verify import verify_bp
    from app.routes.home import home_bp
    from app.routes.represent import represent_bp

    app.register_blueprint(verify_bp)
    app.register_blueprint(home_bp)
    app.register_blueprint(represent_bp)

    # Run the detection function
    detect_device()

    return app

# Detecting if TensorFlow is using GPU
def detect_device():
    devices = tf.config.list_physical_devices()

    print("Available devices:")
    for device in devices:
        print(device)

    if tf.config.list_physical_devices('GPU'):
        print("DeepFace is using a GPU via TensorFlow")
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    else:
        print("DeepFace is using a CPU via TensorFlow")

