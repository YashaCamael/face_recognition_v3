from flask import Flask
import logging
import tensorflow as tf
from google.cloud import storage
from google.cloud.trace import Tracer
from google.cloud.trace.ext.flask.middleware import FlaskMiddleware

def create_app():
    """Initializes and configures the Flask application."""
    
    app = Flask(__name__)

    # --- Tracing Setup ---
    # This automatically integrates with Flask's logging to correlate logs and traces.
    tracer = Tracer()
    FlaskMiddleware(app, tracer)
    
    # --- GCS Client Setup ---
    storage_client = storage.Client()
    app.config['STORAGE_CLIENT'] = storage_client

    # --- Import and register all blueprints ---
    from app.routes.home import home_bp
    from app.routes.embedding import embedding_bp
    from app.routes.antispoofing import antispoof_bp
    
    app.register_blueprint(home_bp)
    app.register_blueprint(embedding_bp)
    app.register_blueprint(antispoof_bp)

    # --- Log hardware status on startup ---
    # This runs after the app is configured and before it starts serving.
    with app.app_context():
        detect_device()

    return app

def detect_device():
    """Detects and prints the available TensorFlow devices."""
    
    logging.info("--- Checking for available TensorFlow devices ---")
    devices = tf.config.list_physical_devices()
    if devices:
        for device in devices:
            logging.info(f"Found device: {device}")
    else:
        logging.warning("No physical devices found by TensorFlow.")

    if tf.config.list_physical_devices('GPU'):
        logging.info("✅ TensorFlow is configured to use a GPU.")
    else:
        logging.info("TensorFlow is configured to use a CPU.")