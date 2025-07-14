from flask import Flask
from google.cloud import storage

# OpenTelemetry Imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter


def create_app():
    """Initializes and configures the Flask application."""
    
    app = Flask(__name__)

    # --- OpenTelemetry Tracing Setup ---
    # Create a processor that sends spans to Google Cloud Trace
    processor = BatchSpanProcessor(CloudTraceSpanExporter())
    
    # The provider is now initialized with the processor
    provider = TracerProvider(active_span_processor=processor)
    trace.set_tracer_provider(provider)

    # Instrument the Flask app automatically
    FlaskInstrumentor().instrument_app(app)
    
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