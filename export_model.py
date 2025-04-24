import tensorflow as tf
import os
import argparse
import json
import glob

def save_model(model_type='transfer', checkpoint_dir='training_checkpoints'):
    """
    Export the model for production use.
    
    Parameters:
    - model_type: Type of model to export ('scratch', 'augmented', or 'transfer')
    - checkpoint_dir: Base directory containing the checkpoints
    
    Returns:
    - Path to the saved model
    """
    # Configuration 
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    IMAGE_CHANNELS = 3
    
    # Set the input shape based on the model configuration
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    
    # Check for checkpoint files to determine the number of classes
    checkpoint_path = os.path.join(checkpoint_dir, model_type)
    
    if model_type == 'transfer':
        # For transfer learning, check the best model file
        checkpoint_file = os.path.join(checkpoint_path, "cp-best.weights.h5")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"No checkpoint file found at {checkpoint_file}")
        
        # Try to determine NUM_CLASSES from the checkpoint file or by other means
        try:
            # Try to load the class count from a metadata file if it exists
            metadata_file = os.path.join(checkpoint_path, "model_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    NUM_CLASSES = metadata.get('num_classes', 100)  # Default to 100 if not found
                    print(f"Loaded NUM_CLASSES={NUM_CLASSES} from metadata file")
            else:
                # Look for class folders in the organized dataset to determine class count
                organized_dataset_path = "organized_cars_dataset"
                train_dir = os.path.join(organized_dataset_path, "train")
                if os.path.exists(train_dir):
                    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
                    NUM_CLASSES = len(class_dirs)
                    print(f"Determined NUM_CLASSES={NUM_CLASSES} from training directory")
                else:
                    # Default to 100 classes if we can't determine it otherwise
                    # This matches the actual saved weights we observed
                    NUM_CLASSES = 100
                    print(f"Using default NUM_CLASSES={NUM_CLASSES}")
                
                # Save this information for future use
                os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump({'num_classes': NUM_CLASSES}, f)
        except Exception as e:
            print(f"Warning: Error determining class count: {e}")
            NUM_CLASSES = 100  # Default to 100 if all else fails
            print(f"Using default NUM_CLASSES={NUM_CLASSES}")
    else:
        # For other models, try to determine from checkpoint or use default
        try:
            # Try loading the class count from metadata
            metadata_file = os.path.join(checkpoint_path, "model_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    NUM_CLASSES = metadata.get('num_classes', 196)
                    print(f"Loaded NUM_CLASSES={NUM_CLASSES} from metadata file")
            else:
                # Default to standard Stanford Cars dataset size
                NUM_CLASSES = 196
                print(f"Using standard NUM_CLASSES={NUM_CLASSES} for Stanford Cars dataset")
        except Exception as e:
            print(f"Warning: Error determining class count: {e}")
            NUM_CLASSES = 196  # Default to full dataset size
            print(f"Using default NUM_CLASSES={NUM_CLASSES}")
    
    print(f"Building model with {NUM_CLASSES} classes...")
    
    # Define the model architecture based on the type
    if model_type == 'scratch' or model_type == 'augmented':
        # Define the CNN model architecture from scratch
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ])
        
        # Compile the model
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
    
    elif model_type == 'transfer':
        # Build the Transfer Learning model based on ResNet50V2
        from tensorflow.keras.applications import ResNet50V2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        
        # Load the ResNet50V2 base model, excluding the top classification layer
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze the layers of the base model
        base_model.trainable = False
        
        # Add custom layers on top
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        
        # Create the final model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'scratch', 'augmented', or 'transfer'")
    
    # Load weights from the checkpoint
    if model_type == 'transfer':
        # For transfer learning, we saved only the best model
        checkpoint_file = os.path.join(checkpoint_path, "cp-best.weights.h5")
        if os.path.exists(checkpoint_file):
            model.load_weights(checkpoint_file)
            print(f"Loaded weights from {checkpoint_file}")
        else:
            raise FileNotFoundError(f"No checkpoint file found at {checkpoint_file}")
    else:
        # For scratch or augmented, find the latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            model.load_weights(latest_checkpoint)
            print(f"Loaded weights from {latest_checkpoint}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    
    # Save the model in TensorFlow SavedModel format
    export_path = f"saved_model/{model_type}"
    # Use model.export() instead of model.save() for SavedModel format
    model.export(export_path)
    print(f"Model exported to {export_path}")
    
    # Also save a list of class names
    try:
        # Look for class folders in the organized dataset
        train_dir = "organized_cars_dataset/train"
        class_names = sorted([d for d in os.listdir(train_dir) 
                              if os.path.isdir(os.path.join(train_dir, d))])
        
        # Save class names to a JSON file
        class_file_path = os.path.join(export_path, "class_names.json")
        with open(class_file_path, 'w') as f:
            json.dump(class_names, f)
        print(f"Saved {len(class_names)} class names to {class_file_path}")
    except Exception as e:
        print(f"Warning: Could not save class names: {e}")
    
    return export_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained model for production.")
    parser.add_argument("--model-type", type=str, default="transfer",
                        choices=["scratch", "augmented", "transfer"],
                        help="Type of model to export")
    parser.add_argument("--checkpoint-dir", type=str, default="training_checkpoints",
                        help="Base directory containing model checkpoints")
    
    args = parser.parse_args()
    save_model(args.model_type, args.checkpoint_dir)