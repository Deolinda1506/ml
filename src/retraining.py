import os
import shutil
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path
import gc

from database import get_database_manager
from preprocessing import ImagePreprocessor
from model import GlaucomaDetectionModel

class RetrainingService:
    def __init__(self, upload_dir="static/uploads", models_dir="models"):
        self.upload_dir = upload_dir
        self.models_dir = models_dir
        self.db_manager = get_database_manager()
        self.preprocessor = ImagePreprocessor()
        self.is_training = False
        self.training_progress = 0
        self.training_status = "idle"
        self.training_error = None
        self.training_details = {}
        
        # Create directories if they don't exist
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    
    def upload_training_data(self, files: List, class_labels: List[str]) -> Dict[str, Any]:
        """Upload and process training data"""
        try:
            uploaded_files = []
            
            for file, class_label in zip(files, class_labels):
                if file.filename:
                    # Create class directory
                    class_dir = os.path.join(self.upload_dir, class_label)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    # Save file
                    file_path = os.path.join(class_dir, file.filename)
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    # Log upload
                    self.db_manager.log_data_upload(
                        file_name=file.filename,
                        file_size=os.path.getsize(file_path),
                        file_type=file.content_type,
                        class_label=class_label
                    )
                    
                    uploaded_files.append({
                        'filename': file.filename,
                        'class': class_label,
                        'size': os.path.getsize(file_path)
                    })
            
            return {
                'success': True,
                'message': f'Successfully uploaded {len(uploaded_files)} files',
                'files': uploaded_files
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error uploading files: {str(e)}',
                'files': []
            }
    
    def check_retraining_trigger(self, min_new_data: int = 10) -> Dict[str, Any]:
        """Check if retraining should be triggered"""
        try:
            # Count new data in upload directory
            new_data_count = self._count_new_data()
            
            # Get last training time
            last_training = self.db_manager.get_last_training_time()
            
            # Check if enough new data or no previous training
            should_retrain = new_data_count >= min_new_data or last_training is None
            
            return {
                'should_retrain': should_retrain,
                'new_data_count': new_data_count,
                'min_required': min_new_data,
                'last_training': last_training.isoformat() if last_training else None,
                'days_since_last_training': (datetime.now() - last_training).days if last_training else None
            }
            
        except Exception as e:
            return {
                'should_retrain': False,
                'error': str(e),
                'new_data_count': 0,
                'min_required': min_new_data
            }
    
    def _count_new_data(self) -> int:
        """Count new data files in upload directory"""
        count = 0
        for class_name in ['glaucoma', 'normal']:
            class_dir = os.path.join(self.upload_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        count += 1
        return count
    
    def start_retraining(self, epochs: int = 30, batch_size: int = 32, 
                        learning_rate: float = 0.001, model_type: str = 'custom') -> Dict[str, Any]:
        """Start retraining process in background"""
        if self.is_training:
            return {
                'success': False,
                'message': 'Training already in progress'
            }
        
        # Check if we have enough data
        trigger_check = self.check_retraining_trigger()
        if not trigger_check['should_retrain']:
            return {
                'success': False,
                'message': f'Not enough new data. Need at least {trigger_check["min_required"]} new images, have {trigger_check["new_data_count"]}'
            }
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._retrain_model,
            args=(epochs, batch_size, learning_rate, model_type)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return {
            'success': True,
            'message': 'Retraining started successfully',
            'training_id': id(training_thread)
        }
    
    def _retrain_model(self, epochs: int, batch_size: int, learning_rate: float, model_type: str):
        """Background retraining process"""
        try:
            self.is_training = True
            self.training_status = "preparing_data"
            self.training_progress = 0
            self.training_error = None
            self.training_details = {}
            
            # Log training start
            training_id = self.db_manager.log_training_start(
                epochs=epochs,
                batch_size=batch_size,
                training_data_size=self._count_new_data(),
                validation_data_size=0  # Will be calculated
            )
            
            # Prepare combined data
            self.training_status = "loading_data"
            self.training_progress = 10
            
            train_images, train_labels, val_images, val_labels = self._prepare_combined_data()
            
            if len(train_images) == 0:
                raise ValueError("No training data available")
            
            # Update training details
            self.training_details = {
                'total_samples': len(train_images),
                'validation_samples': len(val_images),
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_type': model_type
            }
            
            self.training_status = "creating_model"
            self.training_progress = 20
            
            # Create and compile model
            model = GlaucomaDetectionModel()
            model.create_model(model_type)
            model.compile_model(learning_rate=learning_rate)
            
            self.training_status = "training"
            self.training_progress = 30
            
            # Create data generators with enhanced augmentation for larger datasets
            train_generator, val_generator = self.preprocessor.get_data_generators(
                train_images, train_labels, batch_size=batch_size
            )
            
            # Train model
            history = model.train(
                train_generator, val_generator,
                epochs=epochs, batch_size=batch_size
            )
            
            self.training_progress = 80
            self.training_status = "evaluating"
            
            # Evaluate model
            metrics = model.evaluate(val_images, val_labels)
            
            self.training_progress = 90
            self.training_status = "saving"
            
            # Save model with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"glaucoma_model_{timestamp}.h5"
            model_path = os.path.join(self.models_dir, model_filename)
            
            model.save_model(model_path)
            
            # Save model version
            self.db_manager.save_model_version(
                version=f"v{timestamp}",
                model_path=model_path,
                accuracy=metrics['accuracy'],
                description=f"Retrained model with {len(train_images)} samples"
            )
            
            # Log training completion
            self.db_manager.log_training_completion(
                training_id=training_id,
                accuracy=metrics['accuracy'],
                loss=metrics['loss'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                model_path=model_path
            )
            
            self.training_progress = 100
            self.training_status = "completed"
            
            # Cleanup old backups
            self.cleanup_old_backups()
            
            # Clear memory
            del train_images, train_labels, val_images, val_labels
            gc.collect()
            
        except Exception as e:
            self.training_error = str(e)
            self.training_status = "error"
            
            # Log training error
            if 'training_id' in locals():
                self.db_manager.log_training_error(training_id, str(e))
            
        finally:
            self.is_training = False
    
    def _prepare_combined_data(self):
        """Prepare combined training data from original and new data"""
        # Load original training data
        print("Loading original training data...")
        original_train_images, original_train_labels = self.preprocessor.load_dataset('../data/train')
        
        # Load new data from uploads
        print("Loading new training data...")
        new_train_images, new_train_labels = self.preprocessor.load_dataset(self.upload_dir)
        
        # Combine data
        if len(new_train_images) > 0:
            print(f"Combining {len(original_train_images)} original + {len(new_train_images)} new images")
            combined_images = np.concatenate([original_train_images, new_train_images])
            combined_labels = np.concatenate([original_train_labels, new_train_labels])
        else:
            print(f"Using {len(original_train_images)} original images only")
            combined_images = original_train_images
            combined_labels = original_train_labels
        
        # Load test data for validation
        print("Loading test data for validation...")
        test_images, test_labels = self.preprocessor.load_dataset('../data/test')
        
        return combined_images, combined_labels, test_images, test_labels
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'status': self.training_status,
            'progress': self.training_progress,
            'error': self.training_error,
            'details': self.training_details
        }
    
    def get_training_history(self, limit: int = 10) -> List[Dict]:
        """Get training history"""
        return self.db_manager.get_training_history(limit)
    
    def cleanup_old_backups(self, keep_versions: int = 3):
        """Clean up old model versions"""
        try:
            # Get all model files
            model_files = []
            for filename in os.listdir(self.models_dir):
                if filename.startswith('glaucoma_model_') and filename.endswith('.h5'):
                    file_path = os.path.join(self.models_dir, filename)
                    model_files.append((file_path, os.path.getctime(file_path)))
            
            # Sort by creation time (newest first)
            model_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old versions
            for file_path, _ in model_files[keep_versions:]:
                try:
                    os.remove(file_path)
                    print(f"Removed old model: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
                    
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Global retraining service instance
retraining_service = RetrainingService()

def get_retraining_service() -> RetrainingService:
    """Get retraining service instance"""
    return retraining_service

def trigger_retraining(epochs: int = 30, batch_size: int = 32, 
                      learning_rate: float = 0.001, model_type: str = 'custom') -> Dict[str, Any]:
    """Convenience function to trigger retraining"""
    return retraining_service.start_retraining(epochs, batch_size, learning_rate, model_type)

def get_training_status() -> Dict[str, Any]:
    """Convenience function to get training status"""
    return retraining_service.get_training_status() 