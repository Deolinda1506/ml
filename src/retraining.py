import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Ensure eager execution is on

import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .preprocessing import load_data
from .model import calculate_class_weights
from .train_model import train_model

# Model and data paths for glaucoma detection
MODEL_PATH = "models/best_model.h5"
TRAIN_DIR = "/workspaces/ml/data/train"
TEST_DIR = "/workspaces/ml/data/test"
UPLOAD_DIR = "/workspaces/ml/data/new_uploads"
IMAGE_SIZE = (224, 224)  # Updated to match model.py
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2

def preprocess_uploaded_data():
    """
    Preprocess uploaded data and merge with training data.
    """
    print("Preprocessing uploaded data...")
    
    # Check if there are uploaded files
    if not os.path.exists(UPLOAD_DIR):
        print("No uploaded data found.")
        return False
    
    # Get uploaded files from database
    from .database import get_database
    db = get_database()
    uploads = db.get_uploads()
    
    if not uploads:
        print("No uploaded data in database.")
        return False
    
    print(f"Found {len(uploads)} uploaded files")
    
    # Process each uploaded file
    for upload in uploads:
        if upload['status'] == 'pending':
            # Move file to appropriate training directory
            source_path = upload['file_path']
            label = upload['label']
            target_dir = os.path.join(TRAIN_DIR, label)
            
            if os.path.exists(source_path):
                # Ensure target directory exists
                os.makedirs(target_dir, exist_ok=True)
                
                # Move file to training directory
                filename = os.path.basename(source_path)
                target_path = os.path.join(target_dir, filename)
                
                # Avoid overwriting existing files
                if not os.path.exists(target_path):
                    shutil.move(source_path, target_path)
                    print(f"Moved {filename} to {target_dir}")
                else:
                    # If file exists, create unique name
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(target_path):
                        new_filename = f"{base}_{counter}{ext}"
                        target_path = os.path.join(target_dir, new_filename)
                        counter += 1
                    shutil.move(source_path, target_path)
                    print(f"Moved {filename} to {target_path}")
                
                # Update database status
                db.update_upload_status(upload['id'], 'processed')
    
    # Clean up upload directory
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    print("Uploaded data preprocessing completed.")
    return True

def retrain_model():
    """
    Retrains the glaucoma detection model with new data.
    """
    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except:
        print("No existing model found. Please train a new model first.")
        return

    # Preprocess uploaded data
    has_new_data = preprocess_uploaded_data()
    
    if has_new_data:
        print("Retraining with new data...")
    else:
        print("No new data found, retraining with existing data...")

    # Recompile with a new optimizer instance
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )

    print("Loading data...")
    train_gen, val_gen, test_gen, class_indices = load_data(
        TRAIN_DIR, TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT
    )
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights()
    print(f"Class weights: {class_weights}")
    
    # Setup callbacks
    early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', verbose=1)
    
    print("Retraining...")
    
    # Use the train_model function for consistency
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    num_classes = 2  # normal and glaucoma
    
    model, history = train_model(
        train_gen,
        val_gen,
        input_shape,
        num_classes,
        class_weight=class_weights,
        model_path=MODEL_PATH
    )
    
    print("Retraining complete.")
    
    return model, history

if __name__ == "__main__":
    retrain_model() 