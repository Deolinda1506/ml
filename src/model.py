import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset configuration for glaucoma detection
DATASET_PATH = 'data'
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
TEST_PATH = os.path.join(DATASET_PATH, 'test')
CLASSES = ['normal', 'glaucoma']
NUM_CLASSES = len(CLASSES)
INPUT_SHAPE = (224, 224, 3)  # Standard image size for medical imaging

def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Builds a CNN model for glaucoma detection.
    
    Args:
        input_shape (tuple): Shape of input images (224, 224, 3).
        num_classes (int): Number of output classes (2 for normal/glaucoma).

    Returns:
        model (Sequential): Compiled Keras model for glaucoma detection.
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Fourth convolutional block for better feature extraction
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    # Compile for binary classification (normal vs glaucoma)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    
    return model

def get_data_generators(batch_size=32, validation_split=0.2):
    """
    Creates data generators for training and validation.
    
    Args:
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of data to use for validation.
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        subset='training'
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        subset='validation'
    )
    
    return train_generator, val_generator

def get_test_generator(batch_size=32):
    """
    Creates data generator for testing.
    
    Args:
        batch_size (int): Batch size for testing.
        
    Returns:
        test_generator: Test data generator
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )
    
    return test_generator

def calculate_class_weights():
    """
    Calculates class weights to handle imbalanced dataset.
    
    Returns:
        dict: Class weights for training
    """
    # Count images in each class
    class_counts = {}
    for class_name in CLASSES:
        class_path = os.path.join(TRAIN_PATH, class_name)
        if os.path.exists(class_path):
            class_counts[class_name] = len([f for f in os.listdir(class_path) 
                                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Calculate weights
    total_samples = sum(class_counts.values())
    class_weights = {}
    for i, class_name in enumerate(CLASSES):
        class_weights[i] = total_samples / (len(CLASSES) * class_counts.get(class_name, 1))
    
    return class_weights 