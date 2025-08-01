#!/usr/bin/env python3
"""
Glaucoma Detection - Model Training and Evaluation Script
This script contains all the code from the Jupyter notebook for glaucoma detection.

To convert this to a Jupyter notebook:
1. Open Jupyter Lab/Notebook
2. Create a new Python notebook
3. Copy and paste each section into separate cells
4. Add markdown cells for explanations

Or run this script directly for training and evaluation.
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import sys
sys.path.append('../src')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Import our custom modules
from preprocessing import ImagePreprocessor
from model import GlaucomaDetectionModel
from prediction import PredictionService

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

print("\n" + "="*60)
print("1. DATA LOADING AND PREPROCESSING")
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
print(f"Classes: {np.unique(train_labels)}")

# Analyze dataset characteristics
print("\nDataset Analysis:")
analysis = preprocessor.analyze_dataset('../data/train')
print(json.dumps(analysis, indent=2))

# Create visualizations
print("\nCreating dataset visualizations...")
os.makedirs('../static', exist_ok=True)
preprocessor.create_visualizations('../data/train', save_path='../static/dataset_analysis.png')

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

# =============================================================================
# 2. MODEL ARCHITECTURE AND TRAINING
# =============================================================================

print("\n" + "="*60)
print("2. MODEL ARCHITECTURE AND TRAINING")
print("="*60)

# Create and compile model
model = GlaucomaDetectionModel(
    input_shape=(224, 224, 3),
    model_type='custom'
)

# Create model
model.create_model()
model.compile_model(learning_rate=0.001, optimizer_name='adam')

# Display model summary
print("Model Architecture:")
print(model.get_model_summary())

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
    epochs=30,  # Reduced for faster training
    batch_size=batch_size,
    model_save_path='../models/glaucoma_model.h5'
)

print("Training completed!")

# =============================================================================
# 3. MODEL EVALUATION
# =============================================================================

print("\n" + "="*60)
print("3. MODEL EVALUATION")
print("="*60)

# Plot training history
print("Creating training history visualization...")
model.plot_training_history(save_path='../static/training_history.png')

# Evaluate on test set
print("\nEvaluating model on test set...")
test_images_normalized = test_images / 255.0
metrics = model.evaluate(test_images_normalized, test_labels)

print("\nTest Set Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Plot confusion matrix
print("\nCreating confusion matrix...")
model.plot_confusion_matrix(test_images_normalized, test_labels, save_path='../static/confusion_matrix.png')

# Detailed classification report
predictions = model.model.predict(test_images_normalized)
predicted_labels = (predictions > 0.5).astype(int).flatten()

print("\nDetailed Classification Report:")
print(classification_report(test_labels, predicted_labels, target_names=['Normal', 'Glaucoma']))

# =============================================================================
# 4. MODEL PERFORMANCE ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("4. MODEL PERFORMANCE ANALYSIS")
print("="*60)

# Analyze prediction confidence distribution
print("Creating confidence analysis plots...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(predictions[test_labels == 0], bins=20, alpha=0.7, label='Normal', color='green')
plt.hist(predictions[test_labels == 1], bins=20, alpha=0.7, label='Glaucoma', color='red')
plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')
plt.title('Prediction Confidence Distribution')
plt.legend()

plt.subplot(1, 2, 2)
confidence_scores = np.maximum(predictions.flatten(), 1 - predictions.flatten())
plt.hist(confidence_scores, bins=20, alpha=0.7, color='blue')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Model Confidence Distribution')

plt.tight_layout()
plt.savefig('../static/confidence_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve and AUC
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(test_labels, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('../static/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. FEATURE ANALYSIS AND INTERPRETABILITY
# =============================================================================

print("\n" + "="*60)
print("5. FEATURE ANALYSIS AND INTERPRETABILITY")
print("="*60)

# Grad-CAM visualization for model interpretability
def generate_gradcam(model, img, class_index=1):
    """Generate Grad-CAM visualization"""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.layers[-2].output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, class_index]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Visualize attention maps for sample images
print("Creating attention maps...")
sample_indices = np.random.choice(len(test_images), 6, replace=False)

plt.figure(figsize=(15, 10))
for i, idx in enumerate(sample_indices):
    img = test_images[idx:idx+1] / 255.0
    true_label = test_labels[idx]
    pred_prob = predictions[idx][0]
    
    # Generate heatmap
    heatmap = generate_gradcam(model.model, img)
    
    # Resize heatmap to match image size
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (224, 224)).numpy().squeeze()
    
    plt.subplot(2, 3, i+1)
    plt.imshow(test_images[idx])
    plt.imshow(heatmap_resized, alpha=0.6, cmap='jet')
    plt.title(f'True: {true_label}, Pred: {pred_prob:.3f}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('../static/attention_maps.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. MODEL DEPLOYMENT PREPARATION
# =============================================================================

print("\n" + "="*60)
print("6. MODEL DEPLOYMENT PREPARATION")
print("="*60)

# Save model with metadata
print("Saving model...")
model.save_model('../models/glaucoma_model.h5')

# Test prediction service
print("Testing prediction service...")
prediction_service = PredictionService('../models/glaucoma_model.h5')

# Test with sample image
sample_image = test_images[0]
result = prediction_service.predict_single(sample_image)

print("Sample Prediction Test:")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Status: {result['status']}")

# Performance summary
print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("="*50)

# Save performance metrics
performance_summary = {
    'accuracy': float(metrics['accuracy']),
    'precision': float(metrics['precision']),
    'recall': float(metrics['recall']),
    'f1_score': float(metrics['f1_score']),
    'roc_auc': float(roc_auc),
    'training_samples': len(train_images),
    'validation_samples': len(X_val),
    'test_samples': len(test_images),
    'model_parameters': model.model.count_params(),
    'created_at': datetime.now().isoformat()
}

with open('../models/performance_summary.json', 'w') as f:
    json.dump(performance_summary, f, indent=4)

print("Performance summary saved to ../models/performance_summary.json")

# =============================================================================
# 7. MODEL COMPARISON AND OPTIMIZATION
# =============================================================================

print("\n" + "="*60)
print("7. MODEL COMPARISON AND OPTIMIZATION")
print("="*60)

# Test different model architectures
model_types = ['custom', 'vgg16', 'resnet50']
results = {}

for model_type in model_types:
    print(f"\nTesting {model_type.upper()} model...")
    
    try:
        # Create model
        test_model = GlaucomaDetectionModel(
            input_shape=(224, 224, 3),
            model_type=model_type
        )
        test_model.create_model()
        test_model.compile_model()
        
        # Quick evaluation (small subset for speed)
        subset_size = min(100, len(test_images))
        subset_images = test_images[:subset_size] / 255.0
        subset_labels = test_labels[:subset_size]
        
        metrics = test_model.evaluate(subset_images, subset_labels)
        results[model_type] = metrics
        
        print(f"{model_type.upper()} - Accuracy: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error with {model_type}: {str(e)}")
        results[model_type] = None

# Compare model performances
if len(results) > 1:
    print("\nCreating model comparison visualization...")
    plt.figure(figsize=(10, 6))
    
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] if results[name] else 0 for name in model_names]
    
    bars = plt.bar(model_names, accuracies, color=['#667eea', '#764ba2', '#f093fb'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../static/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("TRAINING AND EVALUATION COMPLETED!")
print("="*60)
print(f"Model saved to: ../models/glaucoma_model.h5")
print(f"Performance metrics saved to: ../models/performance_summary.json")
print(f"Visualizations saved to: ../static/")
print("\nNext steps:")
print("1. Start the web application: python src/app.py")
print("2. Access the dashboard at: http://localhost:8000")
print("3. Upload images for prediction")
print("4. Monitor model performance")
print("="*60) 