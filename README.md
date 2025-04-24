# Car Make and Model Recognition API

This project deploys a trained CNN model for recognizing car makes and models as a scalable API service using Docker, Kubernetes, and Google Cloud Platform.

## Project Overview

The Car Make and Model Recognition system uses a deep learning model trained on the Stanford Cars dataset to identify car makes and models from images. The system is deployed as a REST API service that accepts image uploads and returns the predicted car make and model with confidence scores.

## Model Architecture

We use a transfer learning approach with ResNet50V2 as the base model. This provides excellent accuracy by leveraging pre-trained weights from ImageNet while fine-tuning the top layers for our specific car classification task.

## Deployment Architecture

- **Flask API**: Serves the model predictions via HTTP endpoints
- **Docker**: Containerizes the application for consistent deployment
- **Google Kubernetes Engine (GKE)**: Provides scalable, managed Kubernetes
- **Cloud Load Balancing**: Routes traffic to multiple instances
- **Horizontal Pod Autoscaler**: Scales based on CPU utilization

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Docker
- Google Cloud SDK
- kubectl

## Directory Structure

```
.
├── app.py                 # Flask API application
├── Car_Recognition_CNN.ipynb  # Jupyter notebook with model training
├── deploy.sh              # Deployment script for GCP
├── Dockerfile             # Docker container configuration
├── export_model.py        # Script to export the trained model
├── kubernetes/            # Kubernetes configuration files
│   └── deployment.yaml    # K8s deployment, service, and HPA configs
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── saved_model/           # Directory for saved models (created by export_model.py)
└── training_checkpoints/  # Checkpoint files from model training
```

## Deployment Steps

### 1. Export the Model

The first step is to export your trained model to a format suitable for production:

```bash
python export_model.py --model-type transfer
```

This script will:
- Load the model architecture and weights from checkpoints
- Export the model in TensorFlow SavedModel format
- Save class names for prediction mapping

### 2. Build and Test the Docker Image Locally

Build the Docker image:

```bash
docker build -t car-recognition-api:latest .
```

Run the container locally:

```bash
docker run -p 8080:8080 car-recognition-api:latest
```

Test the API:

```bash
curl -X POST -F "file=@/path/to/car/image.jpg" http://localhost:8080/predict
```

### 3. Deploy to Google Cloud

You can either follow the step-by-step process below or use the provided deployment script:

```bash
# Make the script executable first
chmod +x deploy.sh
# Edit the script to set your GCP project ID and preferred region
# Then run
./deploy.sh
```

#### Manual Deployment Steps:

1. **Configure gcloud:**
   ```bash
   gcloud auth login
   gcloud config set project [YOUR_PROJECT_ID]
   ```

2. **Push Docker image to Google Container Registry (GCR):**
   ```bash
   gcloud auth configure-docker
   docker tag car-recognition-api:latest gcr.io/[YOUR_PROJECT_ID]/car-recognition-api:latest
   docker push gcr.io/[YOUR_PROJECT_ID]/car-recognition-api:latest
   ```

3. **Create a GKE cluster:**
   ```bash
   gcloud container clusters create car-recognition-cluster \
       --num-nodes=2 \
       --machine-type=n1-standard-2 \
       --region=us-central1
   ```

4. **Get credentials for the cluster:**
   ```bash
   gcloud container clusters get-credentials car-recognition-cluster --region=us-central1
   ```

5. **Update the Kubernetes deployment file:**
   Edit `kubernetes/deployment.yaml` to replace `[PROJECT_ID]` with your actual GCP project ID.

6. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f kubernetes/deployment.yaml
   ```

7. **Get the external IP:**
   ```bash
   kubectl get service car-recognition-api
   ```

## Using the API

The API exposes the following endpoints:

### Health Check

```
GET /health
```

Returns the health status of the API and model.

### Prediction

```
POST /predict
```

Parameters:
- `file`: An image file upload (JPG, JPEG, or PNG)

Response:
```json
{
  "status": "success",
  "predictions": [
    {"class": "Bugatti_Veyron_16.4_Coupe_2009", "confidence": 98.2},
    {"class": "Bugatti_Veyron_16.4_Convertible_2009", "confidence": 1.5},
    {"class": "Ferrari_458_Italia_Coupe_2012", "confidence": 0.2},
    {"class": "Lamborghini_Aventador_Coupe_2012", "confidence": 0.1},
    {"class": "Ferrari_California_Convertible_2012", "confidence": 0.0}
  ],
  "top_prediction": {
    "class": "Bugatti_Veyron_16.4_Coupe_2009",
    "confidence": 98.2
  }
}
```

## Scaling and Monitoring

- The API automatically scales based on CPU utilization thanks to the Horizontal Pod Autoscaler
- You can monitor the deployment using Google Cloud Console or kubectl:
  ```bash
  kubectl get pods
  kubectl get hpa
  kubectl describe deployment car-recognition-api
  ```

## Cleanup

To avoid incurring charges, delete the resources when not in use:

```bash
# Delete the GKE cluster
gcloud container clusters delete car-recognition-cluster --region=us-central1

# Delete the container image
gcloud container images delete gcr.io/[YOUR_PROJECT_ID]/car-recognition-api:latest
```