import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
from google.cloud import storage

# Create Flask app with model loading on startup
app = Flask(__name__)

# Configuration
MODEL_DIR = os.environ.get('MODEL_DIR', '/tmp/models/transfer')
CLOUD_STORAGE_BUCKET = os.environ.get('CLOUD_STORAGE_BUCKET', 'car-recognition-models-europe')
CLOUD_STORAGE_MODEL_PATH = os.environ.get('CLOUD_STORAGE_MODEL_PATH', 'models/transfer')
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
IMAGE_SIZE = (224, 224)

# Create upload and model folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables for model and class names
model = None
class_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_model_from_gcs():
    """Download model from Google Cloud Storage"""
    try:
        print(f"Downloading model from gs://{CLOUD_STORAGE_BUCKET}/{CLOUD_STORAGE_MODEL_PATH}")
        client = storage.Client()
        bucket = client.get_bucket(CLOUD_STORAGE_BUCKET)
        
        # List all blobs in the model directory
        blobs = list(bucket.list_blobs(prefix=CLOUD_STORAGE_MODEL_PATH))
        
        print(f"Found {len(blobs)} files in bucket")
        for blob in blobs:
            print(f"Found file: {blob.name}")
        
        if not blobs:
            raise Exception(f"No files found in gs://{CLOUD_STORAGE_BUCKET}/{CLOUD_STORAGE_MODEL_PATH}")
        
        # Download each file preserving directory structure
        for blob in blobs:
            destination_path = os.path.join(MODEL_DIR, os.path.relpath(blob.name, CLOUD_STORAGE_MODEL_PATH))
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            print(f"Downloading {blob.name} to {destination_path}")
            blob.download_to_filename(destination_path)
            
        print(f"Model successfully downloaded to {MODEL_DIR}")
        print(f"Checking that files were downloaded correctly:")
        for root, dirs, files in os.walk(MODEL_DIR):
            for file in files:
                print(f"  {os.path.join(root, file)}")
        
        # Specifically check for class names file
        class_path = os.path.join(MODEL_DIR, "class_names.json")
        if os.path.exists(class_path):
            print(f"class_names.json exists at {class_path}, size: {os.path.getsize(class_path)} bytes")
            # Print first few entries to verify content
            with open(class_path, 'r') as f:
                content = f.read()
                print(f"First 200 characters: {content[:200]}...")
        else:
            print(f"ERROR: class_names.json NOT FOUND at {class_path}")
        
        return True
    except Exception as e:
        print(f"Error downloading model from GCS: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def load_model():
    """Load the model and class names at startup"""
    global model, class_names
    
    # Check if we need to download the model from GCS
    if not os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
        success = download_model_from_gcs()
        if not success:
            raise Exception("Failed to download model from Google Cloud Storage")
    
    # Load saved model
    try:
        # Use standard TensorFlow SavedModel loading instead of TFSMLayer
        model = tf.saved_model.load(MODEL_DIR)
        print(f"Model loaded from {MODEL_DIR} using tf.saved_model.load")
        
        # Load class names
        class_file = os.path.join(MODEL_DIR, "class_names.json")
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                class_names = json.load(f)
            print(f"Loaded {len(class_names)} class names")
        else:
            print(f"Warning: Class names file not found at {class_file}")
            # Create dummy class names as a fallback - we'll need to determine this later
            class_names = [f"Class_{i}" for i in range(100)]  # Assuming 100 classes as a safe default
            print(f"Created {len(class_names)} dummy class names")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Initialize the model - removed @app.before_first_request and will call explicitly
# before starting the server

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for car make and model prediction
    
    Expects an image file uploaded with the key 'file'
    Returns prediction results with class names and confidence scores
    """
    # Ensure model is loaded
    global model
    if model is None:
        load_model()
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Process the image
        try:
            # Read image file
            img_bytes = file.read()
            print(f"Read {len(img_bytes)} bytes from uploaded file: {file.filename}")
            
            img = Image.open(io.BytesIO(img_bytes))
            print(f"Opened image: size={img.size}, mode={img.mode}")
            
            # Convert to RGB if needed (in case of PNG with alpha channel)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"Converted image to RGB mode")
            
            # Resize to expected size
            img = img.resize(IMAGE_SIZE)
            print(f"Resized image to {IMAGE_SIZE}")
            
            # Convert to array and preprocess
            img_array = image.img_to_array(img)
            print(f"Converted to array: shape={img_array.shape}, dtype={img_array.dtype}")
            
            img_array = np.expand_dims(img_array, axis=0)
            print(f"Expanded dimensions: shape={img_array.shape}")
            
            # Apply preprocessing based on the model type
            # For ResNet50V2 (transfer learning model)
            img_array = preprocess_input(img_array)
            print(f"Applied preprocessing")
            
            # Make prediction using the saved model's serving signature
            print(f"Running prediction with model...")
            
            # Use the model's serving_default signature
            infer = model.signatures["serving_default"]
            predictions_tensor = infer(tf.convert_to_tensor(img_array))
            
            # Get the output tensor (depends on model output layer name)
            if len(predictions_tensor) == 1:
                # If there's only one output tensor, use that
                first_key = list(predictions_tensor.keys())[0]
                predictions = predictions_tensor[first_key]
                print(f"Using output tensor with key: {first_key}")
            else:
                # Try common output names
                if "predictions" in predictions_tensor:
                    predictions = predictions_tensor["predictions"]
                elif "logits" in predictions_tensor:
                    predictions = predictions_tensor["logits"]
                elif "output_0" in predictions_tensor:
                    predictions = predictions_tensor["output_0"]
                else:
                    # Use the first one as a fallback
                    first_key = list(predictions_tensor.keys())[0]
                    predictions = predictions_tensor[first_key]
                    print(f"Using output tensor with key: {first_key}")
            
            print(f"Prediction completed: shape={predictions.shape}")
            
            # Convert to numpy array for processing
            predictions_np = predictions.numpy()
            top_5_idx = np.argsort(predictions_np[0])[-5:][::-1]
            print(f"Top 5 indices: {top_5_idx}")
            
            # Check if class_names list is long enough to contain all indices
            if max(top_5_idx) >= len(class_names):
                print(f"Warning: Prediction index {max(top_5_idx)} exceeds class_names length {len(class_names)}")
                # Create dummy class names for any missing indices
                extended_class_names = class_names.copy()
                while len(extended_class_names) <= max(top_5_idx):
                    extended_class_names.append(f"Class_{len(extended_class_names)}")
                class_names_to_use = extended_class_names
            else:
                class_names_to_use = class_names
                
            top_5_predictions = [
                {
                    "class": class_names_to_use[idx],
                    "confidence": float(predictions_np[0][idx] * 100)  # Convert to percentage
                }
                for idx in top_5_idx
            ]
            print(f"Formatted predictions: {top_5_predictions}")
            
            # Return results
            return jsonify({
                'status': 'success',
                'predictions': top_5_predictions,
                'top_prediction': {
                    "class": class_names_to_use[top_5_idx[0]],
                    "confidence": float(predictions_np[0][top_5_idx[0]] * 100)
                }
            })
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Error processing image: {str(e)}")
            print(f"Traceback: {error_traceback}")
            return jsonify({'error': f'Error processing image: {str(e)}', 'traceback': error_traceback}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Kubernetes liveness probe"""
    global model
    if model is None:
        try:
            load_model()
            return jsonify({'status': 'healthy', 'model_type': os.path.basename(MODEL_DIR)})
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Failed to load model: {str(e)}'}), 500
    return jsonify({'status': 'healthy', 'model_type': os.path.basename(MODEL_DIR)})

if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)