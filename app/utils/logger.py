from flask import request
import json
import sys

def sanitize_data(data):
    """
    Recursively sanitizes data for logging by:
    1. Truncating large strings (like base64 images).
    2. Ensuring basic serializability (numpy handled by default=str in dumps).
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
        return data

def check_has_error(data):
    """
    Recursively checks if any dictionary in the data contains an 'error' key.
    """
    if isinstance(data, dict):
        if 'error' in data:
            return True
        return any(check_has_error(v) for v in data.values())
    elif isinstance(data, list):
        return any(check_has_error(item) for item in data)
    return False

def log_api_interaction(route_name, request_data, response_data, status_code=200):
    """
    Logs an API interaction in JSON format for Cloud Run.
    Includes trace context for log grouping.
    """
    try:
        # Extract trace ID for Cloud Run log grouping
        trace_header = request.headers.get('X-Cloud-Trace-Context')
        trace_id = None
        if trace_header:
            trace_id = trace_header.split('/')[0]

        # Determine severity: Error if status_code >= 400 OR if response contains "error"
        has_error = check_has_error(response_data)
        severity = "INFO"
        if status_code >= 400 or has_error:
            severity = "ERROR"

        log_entry = {
            "severity": severity,
            "message": f"API Interaction: {route_name} ({status_code})",
            "http_status": status_code,
            "route": route_name,
            "request": sanitize_data(request_data),
            "response": sanitize_data(response_data),
            "has_error_in_body": has_error
        }
        
        if trace_id:
            # GCP uses this field name for trace ID
            log_entry["logging.googleapis.com/trace"] = f"projects/{current_project_id()}/traces/{trace_id}"

        output = sys.stdout if severity == "INFO" else sys.stderr
        # Use default=str to handle numpy arrays or other non-serializable objects
        print(json.dumps(log_entry, default=str), file=output, flush=True)
        
    except Exception as e:
        # Fallback if primary logging fails
        print(json.dumps({
            "severity": "ERROR",
            "message": f"Logging failed for route {route_name}: {str(e)}"
        }, default=str), file=sys.stderr, flush=True)

def current_project_id():
    """Attempts to get the project ID from environment or metadata."""
    import os
    return os.environ.get('GOOGLE_CLOUD_PROJECT', 'unknown-project')
