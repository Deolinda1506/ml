import os
import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import List, Dict, Tuple, Union
import json
from datetime import datetime
import logging

from .model import GlaucomaDetectionModel
from .preprocessing import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionService:
    """Service for handling predictions"""
    
    def __init__(self, model_path: str = 'models/glaucoma_model.h5'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = ImagePreprocessor()
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = GlaucomaDetectionModel()
                self.model.load_model(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def preprocess_image(self, image_data: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Preprocess image for prediction"""
        try:
            if isinstance(image_data, str):
                # Handle file path
                if os.path.exists(image_data):
                    return self.preprocessor.load_and_preprocess_image(image_data)
                else:
                    # Handle base64 encoded image
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    image = image.convert('RGB')
                    image = np.array(image)
                    return self.preprocessor.load_and_preprocess_image_from_array(image)
            
            elif isinstance(image_data, bytes):
                # Handle bytes
                image = Image.open(io.BytesIO(image_data))
                image = image.convert('RGB')
                image = np.array(image)
                return self.preprocessor.load_and_preprocess_image_from_array(image)
            
            elif isinstance(image_data, np.ndarray):
                # Handle numpy array
                return self.preprocessor.load_and_preprocess_image_from_array(image_data)
            
            else:
                raise ValueError("Unsupported image data type")
                
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_single(self, image_data: Union[str, bytes, np.ndarray]) -> Dict:
        """Predict single image"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction
            confidence, class_label = self.model.predict_single(processed_image)
            
            # Prepare response
            result = {
                'prediction': class_label,
                'confidence': float(confidence),
                'probability': float(confidence if class_label == "Glaucoma" else 1 - confidence),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            logger.info(f"Prediction successful: {class_label} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                'prediction': None,
                'confidence': 0.0,
                'probability': 0.0,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def predict_batch(self, image_data_list: List[Union[str, bytes, np.ndarray]]) -> List[Dict]:
        """Predict multiple images"""
        results = []
        
        for i, image_data in enumerate(image_data_list):
            try:
                result = self.predict_single(image_data)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting image {i}: {str(e)}")
                results.append({
                    'prediction': None,
                    'confidence': 0.0,
                    'probability': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': str(e),
                    'image_index': i
                })
        
        return results
    
    def predict_from_file(self, file_path: str) -> Dict:
        """Predict from file path"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            return self.predict_single(file_path)
            
        except Exception as e:
            logger.error(f"Error predicting from file {file_path}: {str(e)}")
            return {
                'prediction': None,
                'confidence': 0.0,
                'probability': 0.0,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def predict_from_upload(self, uploaded_file) -> Dict:
        """Predict from uploaded file"""
        try:
            # Read file content
            file_content = uploaded_file.read()
            
            # Convert to image
            image = Image.open(io.BytesIO(file_content))
            image = image.convert('RGB')
            image_array = np.array(image)
            
            return self.predict_single(image_array)
            
        except Exception as e:
            logger.error(f"Error predicting from upload: {str(e)}")
            return {
                'prediction': None,
                'confidence': 0.0,
                'probability': 0.0,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def get_prediction_stats(self, predictions: List[Dict]) -> Dict:
        """Get statistics from batch predictions"""
        if not predictions:
            return {}
        
        successful_predictions = [p for p in predictions if p['status'] == 'success']
        
        if not successful_predictions:
            return {
                'total_predictions': len(predictions),
                'successful_predictions': 0,
                'failed_predictions': len(predictions),
                'success_rate': 0.0
            }
        
        # Calculate statistics
        total = len(predictions)
        successful = len(successful_predictions)
        failed = total - successful
        
        # Class distribution
        class_counts = {}
        confidences = []
        
        for pred in successful_predictions:
            class_label = pred['prediction']
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
            confidences.append(pred['confidence'])
        
        stats = {
            'total_predictions': total,
            'successful_predictions': successful,
            'failed_predictions': failed,
            'success_rate': successful / total,
            'class_distribution': class_counts,
            'average_confidence': np.mean(confidences) if confidences else 0.0,
            'min_confidence': np.min(confidences) if confidences else 0.0,
            'max_confidence': np.max(confidences) if confidences else 0.0
        }
        
        return stats
    
    def validate_image(self, image_data: Union[str, bytes, np.ndarray]) -> bool:
        """Validate if image data is valid"""
        try:
            if isinstance(image_data, str):
                if os.path.exists(image_data):
                    # Check if it's a valid image file
                    with Image.open(image_data) as img:
                        img.verify()
                    return True
                else:
                    # Try to decode base64
                    try:
                        image_bytes = base64.b64decode(image_data)
                        with Image.open(io.BytesIO(image_bytes)) as img:
                            img.verify()
                        return True
                    except:
                        return False
            
            elif isinstance(image_data, bytes):
                with Image.open(io.BytesIO(image_data)) as img:
                    img.verify()
                return True
            
            elif isinstance(image_data, np.ndarray):
                # Check if array has valid shape
                if len(image_data.shape) == 3 and image_data.shape[2] in [1, 3, 4]:
                    return True
                return False
            
            return False
            
        except Exception:
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {
                'model_loaded': False,
                'model_path': self.model_path,
                'error': 'Model not loaded'
            }
        
        try:
            model_info = {
                'model_loaded': True,
                'model_path': self.model_path,
                'input_shape': self.model.input_shape,
                'model_type': self.model.model_type,
                'total_params': self.model.model.count_params() if self.model.model else 0
            }
            
            if self.model.metrics:
                model_info['metrics'] = self.model.metrics
            
            return model_info
            
        except Exception as e:
            return {
                'model_loaded': True,
                'model_path': self.model_path,
                'error': str(e)
            }

# Global prediction service instance
prediction_service = None

def get_prediction_service() -> PredictionService:
    """Get global prediction service instance"""
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService()
    return prediction_service

def predict_image(image_data: Union[str, bytes, np.ndarray]) -> Dict:
    """Convenience function for single image prediction"""
    service = get_prediction_service()
    return service.predict_single(image_data)

def predict_batch_images(image_data_list: List[Union[str, bytes, np.ndarray]]) -> List[Dict]:
    """Convenience function for batch image prediction"""
    service = get_prediction_service()
    return service.predict_batch(image_data_list) 