# Production Dockerfile for Telecom Antenna Azimuth Estimation Project
FROM ultralytics/ultralytics:latest

# Set working directory inside container
WORKDIR /app

# Install system dependencies for OpenCV and GL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default entrypoint execution
CMD ["python3", "entrypoint/inference.py"]
