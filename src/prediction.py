import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

def load_trained_model(model_path):
    """
    Loads a trained glaucoma detection model.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return load_model(model_path)

def process_image(img_path, target_size=(224, 224)):
    """
    Processes an image for prediction.
    
    Args:
        img_path (str): Path to the image file
        target_size (tuple): Target image size (height, width)
        
    Returns:
        numpy.ndarray: Processed image array
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(model, img_path, class_labels, target_size=(224, 224)):
    """
    Predicts the class of a single image.
    
    Args:
        model: Trained Keras model
        img_path (str): Path to the image file
        class_labels (dict): Mapping of class indices to labels
        target_size (tuple): Target image size
        
    Returns:
        tuple: (predicted_label, confidence)
    """
    img_array = process_image(img_path, target_size)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    return class_labels[predicted_class_index], confidence

def predict_batch(model, img_paths, class_labels, target_size=(224, 224)):
    """
    Predicts classes for a batch of images.
    
    Args:
        model: Trained Keras model
        img_paths (list): List of image file paths
        class_labels (dict): Mapping of class indices to labels
        target_size (tuple): Target image size
        
    Returns:
        list: List of prediction results
    """
    results = []
    for img_path in img_paths:
        try:
            label, confidence = predict_image(model, img_path, class_labels, target_size)
            results.append({'image': img_path, 'label': label, 'confidence': confidence})
        except Exception as e:
            results.append({'image': img_path, 'error': str(e)})
    return results

def predict_from_directory(model, directory_path, class_labels, target_size=(224, 224)):
    """
    Predicts classes for all images in a directory.
    
    Args:
        model: Trained Keras model
        directory_path (str): Path to directory containing images
        class_labels (dict): Mapping of class indices to labels
        target_size (tuple): Target image size
        
    Returns:
        list: List of prediction results
    """
    img_paths = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_paths.append(os.path.join(directory_path, filename))
    
    return predict_batch(model, img_paths, class_labels, target_size)

if __name__ == "__main__":
    # Configuration for glaucoma detection
    model_path = 'models/best_model.h5'
    class_labels = {0: 'Normal', 1: 'Glaucoma'}  # Adapted for glaucoma detection
    
    # Example image paths from your dataset
    img_paths = [
        'data/train/normal/00001_n.jpg',
        'data/train/glaucoma/00001_g.jpg',
        'data/test/normal/00001_n.png',
        'data/test/glaucoma/00001_g.png'
    ]
    
    try:
        model = load_trained_model(model_path)
        results = predict_batch(model, img_paths, class_labels)
        
        for result in results:
            if 'label' in result:
                print(f"Predicted label for {result['image']}: {result['label']} (confidence: {result['confidence']:.2f})")
            else:
                print(f"Error predicting {result['image']}: {result['error']}")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model file exists at the specified path.")
    except Exception as e:
        print(f"Unexpected error: {e}") 