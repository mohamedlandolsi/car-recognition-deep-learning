# PowerShell Deployment Script for Car Recognition API using Cloud Run
# This is a simplified deployment approach that doesn't require a Kubernetes cluster

# Configuration
$PROJECT_ID = "car-recognition-deep-learning"  # Your GCP project ID
$GCP_REGION = "europe-west1"  # Europe region (close to Tunisia)
$SERVICE_NAME = "car-recognition-api"
$IMAGE_NAME = "car-recognition-api"
$CLOUD_STORAGE_BUCKET = "car-recognition-models-europe"

# Set Google Cloud project
Write-Host "Setting Google Cloud project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs if not already enabled
Write-Host "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com

# Build and push Docker image to Google Container Registry
Write-Host "Building and pushing Docker image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME

# Deploy to Cloud Run
Write-Host "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME `
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME `
    --platform managed `
    --region $GCP_REGION `
    --memory 2Gi `
    --cpu 1 `
    --timeout 300 `
    --set-env-vars="MODEL_DIR=/tmp/models/transfer,CLOUD_STORAGE_BUCKET=$CLOUD_STORAGE_BUCKET,CLOUD_STORAGE_MODEL_PATH=models/transfer" `
    --allow-unauthenticated

# Grant the Cloud Run service account access to Cloud Storage
Write-Host "Setting up Cloud Storage permissions..."
$SERVICE_ACCOUNT = "$(gcloud run services describe $SERVICE_NAME --region $GCP_REGION --format='value(serviceAccountEmail)')"
if ($null -ne $SERVICE_ACCOUNT -and $SERVICE_ACCOUNT -ne "") {
    # Grant storage object viewer role
    gcloud projects add-iam-policy-binding $PROJECT_ID `
        --member="serviceAccount:$SERVICE_ACCOUNT" `
        --role="roles/storage.objectViewer"
    Write-Host "Granted storage object viewer role to $SERVICE_ACCOUNT"
} else {
    Write-Host "Warning: Service account for Cloud Run service not found, skipping IAM configuration"
}

# Get the deployed service URL
$SERVICE_URL = $(gcloud run services describe $SERVICE_NAME --region $GCP_REGION --format "value(status.url)")
Write-Host "Deployment completed successfully!"
Write-Host "Your API is available at: $SERVICE_URL"
Write-Host "Test it with: curl -X POST -F 'file=@path/to/car/image.jpg' $SERVICE_URL/predict"