# Base Python image
FROM python:3.10.12

# Set the HOME environment variable to /app to use /app/.deepface/weights as the model path
ENV HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Copy the rest of the application code into the container at /app
COPY . .

# Run the Python script to download the models
RUN python download_models.py

# Command to run the app using Gunicorn
CMD ["python","wsgi.py"]
