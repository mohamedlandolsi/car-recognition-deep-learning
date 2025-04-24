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

app = Flask(__name__)

# Configuration
MODEL_DIR = os.environ.get('MODEL_DIR', 'saved_model/transfer')
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
IMAGE_SIZE = (224, 224)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and class names
model = None
class_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the model and class names at startup"""
    global model, class_names
    
    # Load saved model
    try:
        model = tf.keras.models.load_model(MODEL_DIR)
        print(f"Model loaded from {MODEL_DIR}")
        
        # Load class names
        class_file = os.path.join(MODEL_DIR, "class_names.json")
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                class_names = json.load(f)
            print(f"Loaded {len(class_names)} class names")
        else:
            print(f"Warning: Class names file not found at {class_file}")
            # Create dummy class names as a fallback
            output_dim = model.output_shape[1]
            class_names = [f"Class_{i}" for i in range(output_dim)]
            print(f"Created {len(class_names)} dummy class names based on output dimension")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.before_first_request
def initialize():
    """Initialize the model before the first request"""
    load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for car make and model prediction
    
    Expects an image file uploaded with the key 'file'
    Returns prediction results with class names and confidence scores
    """
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
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert to RGB if needed (in case of PNG with alpha channel)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to expected size
            img = img.resize(IMAGE_SIZE)
            
            # Convert to array and preprocess
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Apply preprocessing based on the model type
            # For ResNet50V2 (transfer learning model)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            predictions = model.predict(img_array)
            
            # Get the top 5 predictions
            top_5_idx = predictions[0].argsort()[-5:][::-1]
            top_5_predictions = [
                {
                    "class": class_names[idx],
                    "confidence": float(predictions[0][idx] * 100)  # Convert to percentage
                }
                for idx in top_5_idx
            ]
            
            # Return results
            return jsonify({
                'status': 'success',
                'predictions': top_5_predictions,
                'top_prediction': {
                    "class": class_names[top_5_idx[0]],
                    "confidence": float(predictions[0][top_5_idx[0]] * 100)
                }
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Kubernetes liveness probe"""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'healthy', 'model_type': os.path.basename(MODEL_DIR)})

if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)