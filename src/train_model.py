#!/usr/bin/env python3
"""
Training script for Glaucoma Detection Model
This script trains the model and saves it for deployment.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add current directory to path
sys.path.append('.')

from preprocessing import ImagePreprocessor
from model import GlaucomaDetectionModel

def main():
    """Main training function"""
    print("="*60)
    print("GLAUCOMA DETECTION MODEL TRAINING")
    print("="*60)
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize preprocessor
    print("\n1. Initializing preprocessor...")
    preprocessor = ImagePreprocessor(img_size=(224, 224))
    
    # Load dataset
    print("\n2. Loading dataset...")
    train_images, train_labels = preprocessor.load_dataset('../data/train')
    test_images, test_labels = preprocessor.load_dataset('../data/test')
    
    print(f"   Training set: {train_images.shape[0]} images")
    print(f"   Test set: {test_images.shape[0]} images")
    print(f"   Image shape: {train_images.shape[1:]}")
    
    # Analyze dataset
    print("\n3. Analyzing dataset...")
    analysis = preprocessor.analyze_dataset('../data/train')
    print(f"   Total images: {analysis['total_images']}")
    print(f"   Class distribution: {analysis['class_distribution']}")
    
    # Create model
    print("\n4. Creating model...")
    model = GlaucomaDetectionModel(
        input_shape=(224, 224, 3),
        model_type='custom'
    )
    model.create_model()
    model.compile_model(learning_rate=0.001, optimizer_name='adam')
    
    print(f"   Model parameters: {model.model.count_params():,}")
    
    # Create data generators
    print("\n5. Creating data generators...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Split training data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        train_images, train_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=train_labels
    )
    
    batch_size = 32
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Validation samples: {X_val.shape[0]}")
    
    # Train model
    print("\n6. Training model...")
    start_time = datetime.now()
    
    history = model.train(
        train_generator, 
        val_generator,
        epochs=30,  # Reduced for faster training
        batch_size=batch_size,
        model_save_path='../models/glaucoma_model.h5'
    )
    
    training_time = datetime.now() - start_time
    print(f"   Training completed in {training_time}")
    
    # Evaluate model
    print("\n7. Evaluating model...")
    test_images_normalized = test_images / 255.0
    metrics = model.evaluate(test_images_normalized, test_labels)
    
    print("\n   Test Set Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Save model and metrics
    print("\n8. Saving model and metrics...")
    model.save_model('../models/glaucoma_model.h5')
    
    # Save performance summary
    import json
    performance_summary = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'training_samples': len(train_images),
        'validation_samples': len(X_val),
        'test_samples': len(test_images),
        'model_parameters': model.model.count_params(),
        'training_time_minutes': training_time.total_seconds() / 60,
        'model_type': 'custom_cnn',
        'created_at': datetime.now().isoformat()
    }
    
    with open('../models/performance_summary.json', 'w') as f:
        json.dump(performance_summary, f, indent=4)
    
    print("   Model saved to: ../models/glaucoma_model.h5")
    print("   Metrics saved to: ../models/performance_summary.json")
    
    # Test prediction
    print("\n9. Testing prediction...")
    from prediction import PredictionService
    
    prediction_service = PredictionService('../models/glaucoma_model.h5')
    sample_image = test_images[0]
    result = prediction_service.predict_single(sample_image)
    
    print(f"   Sample prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Status: {result['status']}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    print(f"Model saved to: ../models/glaucoma_model.h5")
    print("You can now run the application with: python src/app.py")
    print("="*60)

if __name__ == "__main__":
    main() 