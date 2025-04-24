#!/bin/bash
# Script to build and deploy the car recognition API to Google Cloud

# Configuration
PROJECT_ID="your-gcp-project-id"  # Replace with your actual GCP project ID
GCP_REGION="us-central1"  # Replace with your preferred region
IMAGE_NAME="car-recognition-api"
GCR_IMAGE="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"

# Step 1: Export the model (if not already done)
echo "Step 1: Exporting the model..."
python export_model.py --model-type transfer

# Step 2: Build the Docker image
echo "Step 2: Building Docker image..."
docker build -t ${IMAGE_NAME}:latest .

# Step 3: Configure Docker for GCR
echo "Step 3: Configuring Docker for Google Container Registry..."
gcloud auth configure-docker

# Step 4: Tag and push the image to GCR
echo "Step 4: Tagging and pushing image to Google Container Registry..."
docker tag ${IMAGE_NAME}:latest ${GCR_IMAGE}:latest
docker push ${GCR_IMAGE}:latest

# Step 5: Create a GKE cluster (if not already created)
echo "Step 5: Creating GKE cluster..."
gcloud container clusters create car-recognition-cluster \
    --num-nodes=2 \
    --machine-type=n1-standard-2 \
    --region=${GCP_REGION}

# Step 6: Get credentials for the cluster
echo "Step 6: Getting credentials for the cluster..."
gcloud container clusters get-credentials car-recognition-cluster --region=${GCP_REGION}

# Step 7: Update Kubernetes deployment file with correct project ID
echo "Step 7: Updating Kubernetes deployment file..."
sed -i "s/\[PROJECT_ID\]/${PROJECT_ID}/g" kubernetes/deployment.yaml

# Step 8: Apply Kubernetes configurations
echo "Step 8: Applying Kubernetes configurations..."
kubectl apply -f kubernetes/deployment.yaml

# Step 9: Get the external IP (this may take a few minutes to be assigned)
echo "Step 9: Getting external IP (may take a few minutes)..."
echo "Waiting for external IP to be assigned..."
while true; do
    EXTERNAL_IP=$(kubectl get service car-recognition-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -n "$EXTERNAL_IP" ]; then
        break
    fi
    echo -n "."
    sleep 10
done

echo -e "\nDeployment complete!"
echo "Your Car Recognition API is now available at: http://${EXTERNAL_IP}/predict"
echo "To test, use: curl -X POST -F \"file=@/path/to/car/image.jpg\" http://${EXTERNAL_IP}/predict"