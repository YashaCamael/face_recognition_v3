from flask import Blueprint, jsonify

home_bp = Blueprint('home_bp', __name__)

@home_bp.route('/')
def home():
    return "Welcome to the Face Verification API!"

@home_bp.route('/liveness', methods=['GET'])
def liveness():
    return jsonify({"status": "live"}), 200

@home_bp.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200
