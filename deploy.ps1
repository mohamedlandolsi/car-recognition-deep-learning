# PowerShell Deployment Script for Car Recognition API

# Configuration
$PROJECT_ID = "car-recognition-deep-learning"  # Set this to your actual GCP project ID
$GCP_REGION = "europe-west1"  # Your chosen region
$CLUSTER_NAME = "car-recognition-cluster"
$IMAGE_NAME = "car-recognition-api"
$SERVICE_ACCOUNT = "car-recognition-service-account"

# Set Google Cloud project
Write-Host "Setting Google Cloud project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Build and push Docker image to Google Container Registry
Write-Host "Building and pushing Docker image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME

# Create GKE cluster if it doesn't exist
$clusterExists = $false
try {
    gcloud container clusters describe $CLUSTER_NAME --region=$GCP_REGION | Out-Null
    $clusterExists = $true
} catch {
    $clusterExists = $false
}

if (-not $clusterExists) {
    Write-Host "Creating Kubernetes cluster..."
    gcloud container clusters create $CLUSTER_NAME `
        --region=$GCP_REGION `
        --num-nodes=1 `
        --machine-type=n1-standard-2
} else {
    Write-Host "Cluster $CLUSTER_NAME already exists."
}

# Get credentials for the cluster
Write-Host "Getting cluster credentials..."
gcloud container clusters get-credentials $CLUSTER_NAME --region=$GCP_REGION

# Apply Service Account Configuration
Write-Host "Applying Service Account configuration..."
kubectl apply -f kubernetes/service-account.yaml

# Grant storage access to service account
Write-Host "Setting up Cloud Storage permissions..."
# Get the GKE service account email
$GKE_SA = kubectl get serviceaccount $SERVICE_ACCOUNT -o jsonpath='{.metadata.name}' 2>&1
if ($null -ne $GKE_SA -and $GKE_SA -ne "") {
    # Create a fully qualified service account name
    $FULL_GKE_SA = "${GKE_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
    # Grant storage object viewer role
    gcloud projects add-iam-policy-binding $PROJECT_ID `
        --member="serviceAccount:${FULL_GKE_SA}" `
        --role="roles/storage.objectViewer"
    Write-Host "Granted storage object viewer role to $FULL_GKE_SA"
} else {
    Write-Host "Warning: Service account $SERVICE_ACCOUNT not found, skipping IAM configuration"
}

# Apply Kubernetes configuration
Write-Host "Deploying to Kubernetes..."
kubectl apply -f kubernetes/deployment.yaml

# Replace placeholders in Kubernetes configuration
kubectl set image deployment/car-recognition-api car-recognition-api=gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

Write-Host "Deployment completed successfully!"
Write-Host "To check status: kubectl get pods"
Write-Host "To get the service URL: kubectl get service car-recognition-service"