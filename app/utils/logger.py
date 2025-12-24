import json
import sys
import copy

def sanitize_data(data):
    """
    Recursively sanitizes data for logging by:
    1. Truncating large strings (like base64 images).
    2. Ensuring data is JSON serializable.
    """
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if k in ['img_base64'] and isinstance(v, str):
                sanitized[k] = f"[BASE64_DATA_TRUNCATED, len={len(v)}]"
            else:
                sanitized[k] = sanitize_data(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    else:
        # For simplicity, we just return the value. 
        # In a more robust implementation, we might check for JSON serializability.
        return data

def log_api_interaction(route_name, request_data, response_data, status_code=200):
    """
    Logs an API interaction in JSON format for Cloud Run.
    """
    try:
        log_entry = {
            "severity": "INFO" if status_code < 400 else "ERROR",
            "message": f"API Interaction: {route_name}",
            "http_status": status_code,
            "route": route_name,
            "request": sanitize_data(request_data),
            "response": sanitize_data(response_data)
        }
        
        # Printing to stdout (or stderr for errors) is the standard way 
        # for Cloud Run to collect logs.
        output = sys.stdout if status_code < 400 else sys.stderr
        print(json.dumps(log_entry), file=output)
    except Exception as e:
        # Fallback if primary logging fails
        print(json.dumps({
            "severity": "ERROR",
            "message": f"Logging failed: {str(e)}"
        }), file=sys.stderr)
