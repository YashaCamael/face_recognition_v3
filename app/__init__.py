from flask import Flask
from google.cloud import storage


def create_app():
    """Initializes and configures the Flask application."""
    
    app = Flask(__name__)

    # --- GCS Client Setup ---
    storage_client = storage.Client()
    app.config['STORAGE_CLIENT'] = storage_client

    # --- Import and register blueprints ---
    from app.routes.embedding import embedding_bp
    from app.routes.antispoofing import antispoof_bp
    from app.routes.home import home_bp
    
    app.register_blueprint(home_bp)
    app.register_blueprint(embedding_bp)
    app.register_blueprint(antispoof_bp)
    
    return app