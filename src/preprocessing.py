import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, val_dir, test_dir, image_size=(224, 224), batch_size=32):
    """
    Loads image data from train, validation, and test directories using Keras ImageDataGenerator.
    Adapted for glaucoma detection dataset.

    Args:
        train_dir (str): Path to training images (data/train).
        val_dir (str): Path to validation images (data/val or validation split).
        test_dir (str): Path to test images (data/test).
        image_size (tuple): Size to resize images to (default: (224, 224)).
        batch_size (int): Number of images per batch (default: 32).

    Returns:
        train_gen: Training data generator.
        val_gen: Validation data generator.
        test_gen: Test data generator.
        class_indices: Mapping of class names to indices.
    """
    # Basic augmentation for training, only rescaling for val/test
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    class_indices = train_gen.class_indices
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Class indices: {class_indices}")

    return train_gen, val_gen, test_gen, class_indices

def load_data_with_validation_split(train_dir, test_dir, image_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Loads data with validation split from training data (for when no separate validation directory exists).
    
    Args:
        train_dir (str): Path to training images
        test_dir (str): Path to test images
        image_size (tuple): Size to resize images to
        batch_size (int): Number of images per batch
        validation_split (float): Fraction of training data to use for validation
        
    Returns:
        tuple: (train_gen, val_gen, test_gen, class_indices)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Only rescaling for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class indices
    class_indices = train_generator.class_indices
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"Class indices: {class_indices}")
    
    return train_generator, val_generator, test_generator, class_indices

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses a single image for prediction.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target image size (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    from tensorflow.keras.preprocessing import image
    
    # Load and resize image
    img = image.load_img(image_path, target_size=target_size)
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_dataset_info(data_dir):
    """
    Gets information about the dataset structure.
    
    Args:
        data_dir (str): Path to data directory
        
    Returns:
        dict: Dataset information
    """
    info = {}
    
    if os.path.exists(data_dir):
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        info['classes'] = classes
        info['num_classes'] = len(classes)
        
        class_counts = {}
        for class_name in classes:
            class_path = os.path.join(data_dir, class_name)
            files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(files)
        
        info['class_counts'] = class_counts
        info['total_samples'] = sum(class_counts.values())
    
    return info 