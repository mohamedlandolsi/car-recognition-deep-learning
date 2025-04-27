FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directories for model downloads
RUN mkdir -p /tmp/models /tmp/uploads

# Set environment variables
ENV MODEL_DIR=/tmp/models/transfer
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]