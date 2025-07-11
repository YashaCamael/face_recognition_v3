from flask import Flask
import logging
import sys
import os
import tensorflow as tf
from google.cloud import storage
import google.cloud.logging

def create_app():
    """Initializes and configures the Flask application."""
    
    app = Flask(__name__)
    
    # Instantiate Google Cloud clients
    storage_client = storage.Client()
    logging_client = google.cloud.logging.Client()

    # Set up structured logging for Cloud Run compatibility
    logging_client.setup_logging()

    # Store the storage client in app config for access in other parts of the app
    app.config['STORAGE_CLIENT'] = storage_client

    # Import and register the API blueprints
    from app.routes.verify import verify_bp
    from app.routes.home import home_bp
    from app.routes.represent import represent_bp

    app.register_blueprint(verify_bp)
    app.register_blueprint(home_bp)
    app.register_blueprint(represent_bp)

    # Check and log the available hardware (CPU/GPU) on startup
    detect_device()

    return app

def detect_device():
    """Detects and prints the available TensorFlow devices."""
    
    devices = tf.config.list_physical_devices()
    logging.info("Available TensorFlow devices:")
    for device in devices:
        logging.info(device)

    if tf.config.list_physical_devices('GPU'):
        logging.info("TensorFlow is configured to use a GPU.")
    else:
        logging.info("TensorFlow is configured to use a CPU.")