#!/bin/bash

set -e  # Exit on error

# Configuration
PROJECT_ID="car-recognition-deep-learning"  # Set this to your actual GCP project ID
GCP_REGION="europe-west1"  # Your chosen region
CLUSTER_NAME="car-recognition-cluster"
IMAGE_NAME="car-recognition-api"
SERVICE_ACCOUNT="car-recognition-service-account"

# Set Google Cloud project
echo "Setting Google Cloud project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Build and push Docker image to Google Container Registry
echo "Building and pushing Docker image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME

# Create GKE cluster if it doesn't exist
if ! gcloud container clusters describe $CLUSTER_NAME --region=$GCP_REGION &>/dev/null; then
  echo "Creating Kubernetes cluster..."
  gcloud container clusters create $CLUSTER_NAME \
    --region=$GCP_REGION \
    --num-nodes=1 \
    --machine-type=n1-standard-2
else
  echo "Cluster $CLUSTER_NAME already exists."
fi

# Get credentials for the cluster
echo "Getting cluster credentials..."
gcloud container clusters get-credentials $CLUSTER_NAME --region=$GCP_REGION

# Apply Service Account Configuration
echo "Applying Service Account configuration..."
kubectl apply -f kubernetes/service-account.yaml

# Grant storage access to service account
echo "Setting up Cloud Storage permissions..."
# Get the GKE service account email
GKE_SA=$(kubectl get serviceaccount $SERVICE_ACCOUNT -o jsonpath='{.metadata.name}')
if [ -n "$GKE_SA" ]; then
  # Get the namespace
  NAMESPACE=$(kubectl config view --minify --output 'jsonpath={..namespace}')
  # Create a fully qualified service account name
  FULL_GKE_SA="${GKE_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
  # Grant storage object viewer role
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${FULL_GKE_SA}" \
    --role="roles/storage.objectViewer"
  echo "Granted storage object viewer role to $FULL_GKE_SA"
else
  echo "Warning: Service account $SERVICE_ACCOUNT not found, skipping IAM configuration"
fi

# Apply Kubernetes configuration
echo "Deploying to Kubernetes..."
kubectl apply -f kubernetes/deployment.yaml

# Replace placeholders in Kubernetes configuration
kubectl set image deployment/car-recognition-api car-recognition-api=gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

echo "Deployment completed successfully!"
echo "To check status: kubectl get pods"
echo "To get the service URL: kubectl get service car-recognition-service"