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
        # Update the loading method to use TFSMLayer for Keras 3 compatibility
        inputs = tf.keras.Input(shape=(224, 224, 3))
        tfsm_layer = tf.keras.layers.TFSMLayer(
            MODEL_DIR, 
            call_endpoint='serving_default'
        )
        outputs = tfsm_layer(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        print(f"Model loaded from {MODEL_DIR} using TFSMLayer")
        
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
            
            # Make prediction
            print(f"Running prediction with model...")
            predictions_dict = model(img_array)  # TFSMLayer returns a dictionary
            
            # Extract the actual predictions tensor from the dictionary
            # The key might be 'outputs' or a specific endpoint name
            if isinstance(predictions_dict, dict):
                print(f"Prediction returned a dictionary with keys: {list(predictions_dict.keys())}")
                # Try to get predictions from the dictionary - the exact key depends on the model
                if 'outputs' in predictions_dict:
                    predictions = predictions_dict['outputs']
                    print(f"Using 'outputs' key, shape={predictions.shape}")
                else:
                    # If 'outputs' is not found, try the first key
                    first_key = list(predictions_dict.keys())[0]
                    predictions = predictions_dict[first_key]
                    print(f"Using '{first_key}' key, shape={predictions.shape}")
            else:
                # It might already be a tensor
                predictions = predictions_dict
                print(f"Prediction output is not a dictionary")
            
            # Now that we have the prediction tensor, we can proceed as before
            print(f"Prediction completed: shape={predictions.shape}")
            
            # Convert the EagerTensor to numpy array before using argsort
            predictions_np = predictions.numpy()
            top_5_idx = predictions_np[0].argsort()[-5:][::-1]
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
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'healthy', 'model_type': os.path.basename(MODEL_DIR)})

if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)