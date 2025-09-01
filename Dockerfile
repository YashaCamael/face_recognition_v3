# Base Python image
# FROM asia-southeast2-docker.pkg.dev/subsidi-tepat-bbm-ai/tensorflow-base/tensorflow:2.16.1-gpu
# FROM asia-southeast2-docker.pkg.dev/subsidi-tepat-bbm-ai-dev/ml-image/tensorflow:2.16.1-gpu
FROM tensorflow/tensorflow:2.14.0-gpu

# Set the HOME environment variable to /app to use /app/.deepface/weights as the model path
ENV HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the aorking directory to /app
WORKDIR /app

# Copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --ignore-installed -r requirements.txt

# Install Gunicorn
# RUN pip install gunicorn

# Copy the rest of the application code into the container at /app
COPY . .

# Run the Python script to download the models
RUN python download_models.py

# Command to run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--threads", "4", "--timeout", "900", "wsgi:application"]
# CMD ["python","wsgi.py"]
