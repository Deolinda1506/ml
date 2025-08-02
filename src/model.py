import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
from datetime import datetime

class GlaucomaDetectionModel:
    def __init__(self, img_size=(224, 224), num_classes=2):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_custom_model(self):
        """Build a custom CNN model"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    

    
    def create_model(self, model_type='custom', **kwargs):
        """Create model based on type"""
        if model_type == 'custom':
            self.model = self.build_custom_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """Compile the model"""
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def get_callbacks(self, model_save_path=None, patience=10):
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if model_save_path:
            callbacks.append(
                ModelCheckpoint(
                    model_save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=50, batch_size=32, 
              model_save_path=None, callbacks=None):
        """Train the model"""
        if callbacks is None:
            callbacks = self.get_callbacks(model_save_path)
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_images, test_labels):
        """Evaluate the model"""
        # Convert labels to categorical
        from tensorflow.keras.utils import to_categorical
        test_labels_cat = to_categorical(test_labels, num_classes=self.num_classes)
        
        # Predict
        predictions = self.model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            test_images, test_labels_cat, verbose=0
        )
        
        # Additional metrics
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        auc_score = roc_auc_score(test_labels, predictions[:, 1])
        
        # Classification report
        class_names = ['Normal', 'Glaucoma']
        report = classification_report(
            test_labels, predicted_classes, 
            target_names=class_names, output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predicted_classes)
        
        metrics = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': f1_score,
            'auc_score': auc_score,
            'loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'predicted_classes': predicted_classes.tolist()
        }
        
        return metrics
    
    def predict_single(self, image):
        """Predict single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'probabilities': prediction[0].tolist()
        }
    
    def save_model(self, filepath):
        """Save the model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        
        # Save training history if available
        if self.history is not None:
            history_file = filepath.replace('.h5', '_history.json')
            with open(history_file, 'w') as f:
                json.dump(self.history.history, f, indent=4)
    
    def load_model(self, filepath):
        """Load the model"""
        self.model = keras.models.load_model(filepath)
        return self.model
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, test_labels, predicted_classes, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(test_labels, predicted_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Glaucoma'],
                   yticklabels=['Normal', 'Glaucoma'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            return "No model available"
        
        # Capture model summary
        from io import StringIO
        summary_io = StringIO()
        self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        summary = summary_io.getvalue()
        summary_io.close()
        
        return summary

 