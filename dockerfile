# Use Python 3.12 as the base image
FROM python:3.12-slim

# Install system dependencies needed for OpenCV/Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt .

# Install PyTorch CPU version first (to save space and build time)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Command to run the app
CMD ["uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "7860"]