from flask import Flask
import os
from app.services.face_verification import verify_faces

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

def create_app():
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

    return app
