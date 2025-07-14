from app import create_app

# Create an instance of the Flask application
application = create_app()

if __name__ == "__main__":
    # If this file is run directly, start a simple development server
    application.run(host='0.0.0.0', port=8080)
