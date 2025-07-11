from app import create_app
import logging
import sys
import deepface
import tensorflow

# --- Log Library Versions for Debugging ---
# This ensures version info is logged as soon as the script is run.
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='[%(levelname)s] %(message)s')

logging.info("--- Checking Library Versions ---")
logging.info(f"DeepFace Version: {deepface.__version__}")
logging.info(f"TensorFlow Version: {tensorflow.__version__}")
logging.info("-----------------------------")


# --- Create and Load the Flask App ---
# This imports your app factory from app/__init__.py

# Create an instance of the Flask application
application = create_app()

if __name__ == "__main__":
    # If this file is run directly, start a simple development server
    application.run(host='0.0.0.0', port=8080)
