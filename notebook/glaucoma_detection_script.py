#!/usr/bin/env python3
"""
Glaucoma Detection - Complete Workflow Script
Updated to include Model Loading, Architecture + Retraining, Model Evaluation, 
Plotting, Confusion Matrix, and Classification Report.
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import sys
sys.path.append('../src')

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Import our custom modules
from preprocessing import ImagePreprocessor
from model import GlaucomaDetectionModel
from prediction import PredictionService
from mongo_to_sqlite_converter import GlaucomaPredictor
from retraining import get_retraining_service

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# =============================================================================
# 1. INITIAL TRAINING
# =============================================================================

print("\n" + "="*60)
print("1. INITIAL TRAINING")
print("="*60)

# Initialize preprocessor
preprocessor = ImagePreprocessor(img_size=(224, 224))

# Load dataset
print("Loading dataset...")
train_images, train_labels = preprocessor.load_dataset('../data/train')
test_images, test_labels = preprocessor.load_dataset('../data/test')

print(f"Training set: {train_images.shape[0]} images")
print(f"Test set: {test_images.shape[0]} images")
print(f"Image shape: {train_images.shape[1:]}")

# Split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_labels
)

print(f"\nData splits:")
print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")
print(f"Test set: {test_images.shape[0]} images")

# Create and compile model
model = GlaucomaDetectionModel(img_size=(224, 224), num_classes=2)
model.create_model(model_type='custom')
model.compile_model(learning_rate=0.001, optimizer='adam')

print(f"Model parameters: {model.model.count_params():,}")

# Create data generators with augmentation
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

# Create generators
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

# Train the model
print("\nStarting model training...")
os.makedirs('../models', exist_ok=True)
history = model.train(
    train_generator, 
    val_generator,
    epochs=50,  # More epochs for larger dataset
    batch_size=batch_size,
    model_save_path='../models/glaucoma_model.h5'
)

print("Training completed!")

# =============================================================================
# PRINT TRAINING RESULTS
# =============================================================================

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

# =============================================================================
# 2. MODEL LOADING
# =============================================================================

print("\n" + "="*60)
print("2. MODEL LOADING")
print("="*60)

# Initialize predictor with trained model
print("Loading trained model...")
predictor = GlaucomaPredictor("models/glaucoma_model.h5")

if predictor.model is not None:
    print("✅ Model loaded successfully!")
    print(f"Model parameters: {predictor.model.count_params():,}")
else:
    print("❌ Model loading failed!")

# Test model loading with sample prediction
if os.path.exists("data/test/glaucoma/00001_g.png"):
    print("\nTesting model with sample image...")
    sample_result = predictor.predict_from_file("data/test/glaucoma/00001_g.png")
    print(f"Sample prediction: {sample_result}")

# =============================================================================
# 3. MODEL ARCHITECTURE + RETRAINING
# =============================================================================

print("\n" + "="*60)
print("3. MODEL ARCHITECTURE + RETRAINING")
print("="*60)

# Display model architecture
if predictor.model is not None:
    print("Model Architecture Summary:")
    print(predictor.model.summary())
    
    # Model architecture visualization
    from tensorflow.keras.utils import plot_model
    
    try:
        plot_model(predictor.model, to_file='../static/model_architecture.png', 
                  show_shapes=True, show_layer_names=True)
        print("✅ Model architecture saved to ../static/model_architecture.png")
    except Exception as e:
        print(f"Could not save model architecture: {e}")

# Retraining section
print("\n" + "-" * 40)
print("RETRAINING WITH NEW DATA")
print("-" * 40)

# Initialize retraining service
retraining_service = get_retraining_service()

# Check if retraining is needed
print("Checking retraining trigger...")
trigger_check = retraining_service.check_retraining_trigger(min_new_data=5)
print(f"Should retrain: {trigger_check['should_retrain']}")
print(f"New data count: {trigger_check['new_data_count']}")

if trigger_check['should_retrain']:
    print("\nStarting retraining process...")
    
    # Start retraining with updated parameters for larger dataset
    result = retraining_service.start_retraining(
        epochs=50,  # More epochs for larger dataset
        batch_size=32,  # Larger batch size
        learning_rate=0.001,
        model_type='custom'
    )
    
    if result['success']:
        print("✅ Retraining started successfully!")
        print(f"Training ID: {result['training_id']}")
        
        # Monitor training progress
        print("\nMonitoring training progress...")
        import time
        
        while True:
            status = retraining_service.get_training_status()
            print(f"Status: {status['status']}, Progress: {status['progress']}%")
            
            if status['status'] == 'completed':
                print("✅ Retraining completed!")
                break
            elif status['status'] == 'error':
                print(f"❌ Retraining failed: {status['error']}")
                break
                
            time.sleep(10)  # Check every 10 seconds
        
        # Get training history
        print("\nTraining History:")
        history = retraining_service.get_training_history(limit=5)
        for training in history:
            print(f"Training {training['id']}: {training['status']} - Accuracy: {training.get('accuracy', 'N/A')}")
    
    else:
        print(f"❌ Retraining failed to start: {result['message']}")

else:
    print("No retraining needed at this time.")

# =============================================================================
# 4. MODEL EVALUATION
# =============================================================================

print("\n" + "="*60)
print("4. MODEL EVALUATION")
print("="*60)

# Load test dataset for evaluation
print("Loading test dataset for evaluation...")
test_images, test_labels = predictor.preprocessor.load_dataset("data/test")

if len(test_images) > 0:
    print(f"Loaded {len(test_images)} test images")
    
    # Evaluate model
    print("\nEvaluating model performance...")
    eval_metrics = predictor.evaluate_model(test_images, test_labels)
    
    if 'error' not in eval_metrics:
        print("\n✅ Model evaluation completed!")
        print(f"Test Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"Test Precision: {eval_metrics['precision']:.4f}")
        print(f"Test Recall: {eval_metrics['recall']:.4f}")
        print(f"Test F1-Score: {eval_metrics['f1_score']:.4f}")
    else:
        print(f"❌ Evaluation failed: {eval_metrics['error']}")
else:
    print("❌ No test data found for evaluation")

# =============================================================================
# 5. PLOTTING AND VISUALIZATION
# =============================================================================

print("\n" + "="*60)
print("5. PLOTTING AND VISUALIZATION")
print("="*60)

# Create comprehensive visualizations
print("Creating model performance visualizations...")

# 1. Training History Plot (if available)
if hasattr(predictor, 'history') and predictor.history is not None:
    print("Creating training history plot...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(predictor.history.history['accuracy'], label='Training Accuracy')
    plt.plot(predictor.history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(predictor.history.history['loss'], label='Training Loss')
    plt.plot(predictor.history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(predictor.history.history['precision'], label='Training Precision')
    plt.plot(predictor.history.history['val_precision'], label='Validation Precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../static/training_history_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2. Prediction Confidence Distribution
print("Creating prediction confidence distribution...")
if len(test_images) > 0:
    predictions = predictor.model.predict(test_images / 255.0, verbose=0)
    confidence_scores = np.max(predictions, axis=1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidence_scores[test_labels == 0], bins=20, alpha=0.7, label='Normal', color='green')
    plt.hist(confidence_scores[test_labels == 1], bins=20, alpha=0.7, label='Glaucoma', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(confidence_scores, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Overall Confidence Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../static/confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 6. CONFUSION MATRIX
# =============================================================================

print("\n" + "="*60)
print("6. CONFUSION MATRIX")
print("="*60)

# Create confusion matrix with proper labels
print("Creating confusion matrix...")

if len(test_images) > 0:
    # Get predictions
    predictions = predictor.model.predict(test_images / 255.0, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes)
    
    # Plot confusion matrix with proper labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Glaucoma'],
               yticklabels=['Normal', 'Glaucoma'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('../static/confusion_matrix_updated.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Confusion matrix saved to ../static/confusion_matrix_updated.png")
    
    # Print confusion matrix values
    print("\nConfusion Matrix Values:")
    print("                    Predicted")
    print("                    Normal  Glaucoma")
    print(f"Actual Normal      {cm[0,0]:>6}  {cm[0,1]:>8}")
    print(f"      Glaucoma     {cm[1,0]:>6}  {cm[1,1]:>8}")

# =============================================================================
# 7. CLASSIFICATION REPORT
# =============================================================================

print("\n" + "="*60)
print("7. CLASSIFICATION REPORT")
print("="*60)

# Generate detailed classification report
print("Generating classification report...")

if len(test_images) > 0:
    # Get predictions
    predictions = predictor.model.predict(test_images / 255.0, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Generate classification report
    report = classification_report(test_labels, predicted_classes, 
                                 target_names=['Normal', 'Glaucoma'],
                                 output_dict=True)
    
    print("\nDetailed Classification Report:")
    print("=" * 50)
    print(classification_report(test_labels, predicted_classes, 
                              target_names=['Normal', 'Glaucoma']))
    
    # Save classification report
    with open('../static/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("\n✅ Classification report saved to ../static/classification_report.json")
    
    # Create visualization of classification metrics
    print("\nCreating classification metrics visualization...")
    
    # Extract metrics for visualization
    metrics_data = {
        'Normal': {
            'Precision': report['Normal']['precision'],
            'Recall': report['Normal']['recall'],
            'F1-Score': report['Normal']['f1-score']
        },
        'Glaucoma': {
            'Precision': report['Glaucoma']['precision'],
            'Recall': report['Glaucoma']['recall'],
            'F1-Score': report['Glaucoma']['f1-score']
        }
    }
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Normal class metrics
    normal_metrics = list(metrics_data['Normal'].values())
    normal_labels = list(metrics_data['Normal'].keys())
    ax1.bar(normal_labels, normal_metrics, color=['#2E8B57', '#3CB371', '#66CDAA'])
    ax1.set_title('Normal Class Metrics')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(normal_metrics):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Glaucoma class metrics
    glaucoma_metrics = list(metrics_data['Glaucoma'].values())
    glaucoma_labels = list(metrics_data['Glaucoma'].keys())
    ax2.bar(glaucoma_labels, glaucoma_metrics, color=['#DC143C', '#FF6347', '#FF7F50'])
    ax2.set_title('Glaucoma Class Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(glaucoma_metrics):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../static/classification_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Classification metrics visualization saved to ../static/classification_metrics.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("COMPLETE WORKFLOW FINISHED!")
print("="*60)
print("✅ Initial Training: Completed")
print("✅ Model Loading: Completed")
print("✅ Architecture + Retraining: Completed")
print("✅ Model Evaluation: Completed")
print("✅ Plotting and Visualization: Completed")
print("✅ Confusion Matrix: Completed")
print("✅ Classification Report: Completed")
print("\nFiles saved:")
print("- ../static/model_architecture.png")
print("- ../static/training_history_detailed.png")
print("- ../static/confidence_distribution.png")
print("- ../static/confusion_matrix_updated.png")
print("- ../static/classification_report.json")
print("- ../static/classification_metrics.png")
print("\nNext steps:")
print("1. Start the web application: python src/app.py")
print("2. Access the dashboard at: http://localhost:8000")
print("3. Use the updated predictor for new predictions")
print("4. Monitor model performance with the new analytics")
print("="*60) 