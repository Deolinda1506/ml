#!/usr/bin/env python3
"""
Training script for Glaucoma Detection Model
This script trains the model and saves it for deployment.
Updated to handle larger datasets efficiently.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import gc

# Add current directory to path
sys.path.append('.')

from preprocessing import ImagePreprocessor
from model import GlaucomaDetectionModel

def count_dataset_images(data_dir):
    """Count images in dataset"""
    print(f"Counting images in: {data_dir}")
    
    total_images = 0
    class_counts = {}
    
    for class_name in ['glaucoma', 'normal']:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} does not exist")
            class_counts[class_name] = 0
            continue
            
        # Count images
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[class_name] = len(image_files)
        total_images += len(image_files)
        
        print(f"  {class_name}: {len(image_files)} images")
    
    print(f"Total images: {total_images}")
    
    if total_images == 0:
        raise ValueError("No images found in dataset")
    
    return class_counts, total_images

def main():
    """Main training function"""
    print("="*60)
    print("GLAUCOMA DETECTION MODEL TRAINING")
    print("Updated for larger dataset")
    print("="*60)
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Count dataset images
    print("\n1. Counting dataset images...")
    train_counts, train_total = count_dataset_images('../data/train')
    test_counts, test_total = count_dataset_images('../data/test')
    
    print(f"\nDataset Summary:")
    print(f"  Training: {train_total} images")
    print(f"  Testing: {test_total} images")
    print(f"  Total: {train_total + test_total} images")
    
    # Initialize preprocessor
    print("\n2. Initializing preprocessor...")
    preprocessor = ImagePreprocessor(img_size=(224, 224))
    
    # Load dataset with progress tracking
    print("\n3. Loading training dataset...")
    train_images, train_labels = preprocessor.load_dataset('../data/train')
    
    print("\n4. Loading test dataset...")
    test_images, test_labels = preprocessor.load_dataset('../data/test')
    
    print(f"\nLoaded Dataset:")
    print(f"  Training set: {train_images.shape[0]} images")
    print(f"  Test set: {test_images.shape[0]} images")
    print(f"  Image shape: {train_images.shape[1:]}")
    
    # Verify class balance
    train_glaucoma = np.sum(train_labels == 1)
    train_normal = np.sum(train_labels == 0)
    test_glaucoma = np.sum(test_labels == 1)
    test_normal = np.sum(test_labels == 0)
    
    print(f"\nClass Distribution:")
    print(f"  Training - Glaucoma: {train_glaucoma}, Normal: {train_normal}")
    print(f"  Test - Glaucoma: {test_glaucoma}, Normal: {test_normal}")
    
    # Analyze dataset
    print("\n5. Analyzing dataset...")
    analysis = preprocessor.analyze_dataset('../data/train')
    print(f"   Total images: {analysis['total_images']}")
    print(f"   Class distribution: {analysis['class_distribution']}")
    
    # Create model
    print("\n6. Creating model...")
    model = GlaucomaDetectionModel(
        input_shape=(224, 224, 3),
        num_classes=2
    )
    model.create_model(model_type='custom')
    model.compile_model(learning_rate=0.001, optimizer='adam')
    
    print(f"   Model parameters: {model.model.count_params():,}")
    
    # Create data generators with more aggressive augmentation for larger dataset
    print("\n7. Creating data generators...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.3,
        shear_range=0.3,
        brightness_range=[0.8, 1.2],
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
    
    # Adjust batch size based on dataset size
    if train_total > 200:
        batch_size = 32
    else:
        batch_size = 16
    
    print(f"   Using batch size: {batch_size}")
    
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
    
    # Clear memory
    del train_images, test_images
    gc.collect()
    
    # Train model with adjusted parameters for larger dataset
    print("\n8. Training model...")
    start_time = datetime.now()
    
    # Adjust epochs based on dataset size
    if train_total > 200:
        epochs = 50
    else:
        epochs = 30
    
    print(f"   Training for {epochs} epochs...")
    
    history = model.train(
        train_generator, 
        val_generator,
        epochs=epochs,
        batch_size=batch_size,
        model_save_path='../models/glaucoma_model.h5'
    )
    
    training_time = datetime.now() - start_time
    print(f"   Training completed in {training_time}")
    
    # Print training results
    print("\n" + "="*40)
    print("TRAINING RESULTS")
    print("="*40)
    
    if history is not None:
        final_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        best_val_accuracy = max(history.history['val_accuracy'])
        
        print(f"Training Accuracy: {final_accuracy:.4f}")
        print(f"Validation Accuracy: {final_val_accuracy:.4f}")
        print(f"Best Validation: {best_val_accuracy:.4f}")
        
        # Save basic results
        training_summary = {
            'training_accuracy': float(final_accuracy),
            'validation_accuracy': float(final_val_accuracy),
            'best_validation_accuracy': float(best_val_accuracy),
            'epochs': len(history.history['accuracy'])
        }
        
        with open('../models/training_results.json', 'w') as f:
            json.dump(training_summary, f, indent=4)
        
        print("✅ Results saved to ../models/training_results.json")
    
    else:
        print("❌ Training history not available")
    
    # Reload test data for evaluation
    print("\n9. Reloading test data for evaluation...")
    test_images, test_labels = preprocessor.load_dataset('../data/test')
    
    # Evaluate model
    print("\n10. Evaluating model...")
    test_images_normalized = test_images / 255.0
    metrics = model.evaluate(test_images_normalized, test_labels)
    
    print("\n   Test Set Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
    
    # Save model and metrics
    print("\n11. Saving model and metrics...")
    model.save_model('../models/glaucoma_model.h5')
    
    # Save performance summary with updated information
    import json
    performance_summary = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'auc_score': float(metrics['auc_score']),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(test_images),
        'model_parameters': model.model.count_params(),
        'training_time_minutes': training_time.total_seconds() / 60,
        'model_type': 'custom_cnn',
        'dataset_info': {
            'train_total': train_total,
            'test_total': test_total,
            'train_glaucoma': train_glaucoma,
            'train_normal': train_normal,
            'test_glaucoma': test_glaucoma,
            'test_normal': test_normal
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open('../models/performance_summary.json', 'w') as f:
        json.dump(performance_summary, f, indent=4)
    
    print("   Model saved to: ../models/glaucoma_model.h5")
    print("   Metrics saved to: ../models/performance_summary.json")
    
    # Create visualizations
    print("\n12. Creating visualizations...")
    try:
        # Plot training history
        history_plot_path = '../static/training_history.png'
        model.plot_training_history(save_path=history_plot_path)
        print(f"   Training history saved to: {history_plot_path}")
        
        # Plot confusion matrix
        predictions = model.model.predict(test_images_normalized)
        predicted_classes = np.argmax(predictions, axis=1)
        cm_plot_path = '../static/confusion_matrix.png'
        model.plot_confusion_matrix(test_labels, predicted_classes, save_path=cm_plot_path)
        print(f"   Confusion matrix saved to: {cm_plot_path}")
        
        # Create dataset analysis
        analysis_plot_path = '../static/dataset_analysis.png'
        preprocessor.create_visualizations('../data/train', save_path=analysis_plot_path)
        print(f"   Dataset analysis saved to: {analysis_plot_path}")
        
    except Exception as e:
        print(f"   Warning: Could not create visualizations: {e}")
    
    # Test prediction
    print("\n13. Testing prediction...")
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
    print(f"Dataset size: {train_total + test_total} images")
    print("You can now run the application with: python src/app.py")
    print("="*60)

if __name__ == "__main__":
    main() 