import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from tqdm import tqdm
import gc

class ImagePreprocessor:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Load image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def load_and_preprocess_image_from_array(self, image_array):
        """Preprocess image from numpy array"""
        try:
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 4:  # RGBA
                    # Convert RGBA to RGB
                    image_array = image_array[:, :, :3]
                elif image_array.shape[2] == 1:  # Grayscale
                    # Convert grayscale to RGB
                    image_array = np.stack([image_array[:, :, 0]] * 3, axis=-1)
            
            # Resize image
            img = cv2.resize(image_array, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error preprocessing image array: {e}")
            return None
    
    def load_dataset(self, data_dir, show_progress=True):
        """Load dataset from directory structure with progress tracking"""
        images = []
        labels = []
        
        # Define class mapping
        class_mapping = {'glaucoma': 1, 'normal': 0}
        
        # First, count total images for progress bar
        total_images = 0
        for class_name in class_mapping.keys():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                total_images += len(image_files)
        
        if total_images == 0:
            print(f"Warning: No images found in {data_dir}")
            return np.array([]), np.array([])
        
        # Load images with progress tracking
        processed_images = 0
        if show_progress:
            pbar = tqdm(total=total_images, desc=f"Loading {os.path.basename(data_dir)}")
        
        for class_name, class_id in class_mapping.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for filename in image_files:
                image_path = os.path.join(class_dir, filename)
                img = self.load_and_preprocess_image(image_path)
                if img is not None:
                    images.append(img)
                    labels.append(class_id)
                
                processed_images += 1
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        if len(images) == 0:
            print(f"Warning: No valid images loaded from {data_dir}")
            return np.array([]), np.array([])
        
        print(f"Successfully loaded {len(images)} images from {data_dir}")
        return np.array(images), np.array(labels)
    
    def load_dataset_in_batches(self, data_dir, batch_size=1000, show_progress=True):
        """Load dataset in batches to handle very large datasets"""
        all_images = []
        all_labels = []
        
        # Define class mapping
        class_mapping = {'glaucoma': 1, 'normal': 0}
        
        # Collect all image paths first
        image_paths = []
        labels = []
        
        for class_name, class_id in class_mapping.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for filename in image_files:
                image_path = os.path.join(class_dir, filename)
                image_paths.append(image_path)
                labels.append(class_id)
        
        if len(image_paths) == 0:
            print(f"Warning: No images found in {data_dir}")
            return np.array([]), np.array([])
        
        # Process in batches
        if show_progress:
            pbar = tqdm(total=len(image_paths), desc=f"Loading {os.path.basename(data_dir)}")
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            batch_images = []
            for image_path in batch_paths:
                img = self.load_and_preprocess_image(image_path)
                if img is not None:
                    batch_images.append(img)
                
                if show_progress:
                    pbar.update(1)
            
            if batch_images:
                all_images.extend(batch_images)
                all_labels.extend(batch_labels[:len(batch_images)])
            
            # Clear memory
            del batch_images
            gc.collect()
        
        if show_progress:
            pbar.close()
        
        print(f"Successfully loaded {len(all_images)} images from {data_dir}")
        return np.array(all_images), np.array(all_labels)
    
    def create_data_generators(self, train_images, train_labels, validation_split=0.2):
        """Create data generators with augmentation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(validation_split=validation_split)
        
        return train_datagen, val_datagen
    
    def get_data_generators(self, train_images, train_labels, batch_size=32, validation_split=0.2):
        """Get data generators for training and validation"""
        train_datagen, val_datagen = self.create_data_generators(
            train_images, train_labels, validation_split
        )
        
        # Convert labels to categorical
        from tensorflow.keras.utils import to_categorical
        train_labels_cat = to_categorical(train_labels, num_classes=2)
        
        # Create generators
        train_generator = train_datagen.flow(
            train_images, train_labels_cat,
            batch_size=batch_size,
            subset='training'
        )
        
        val_generator = val_datagen.flow(
            train_images, train_labels_cat,
            batch_size=batch_size,
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def analyze_dataset(self, data_dir):
        """Analyze dataset statistics"""
        analysis = {
            'total_images': 0,
            'class_distribution': {},
            'image_sizes': [],
            'file_formats': {},
            'corrupted_images': 0
        }
        
        for class_name in ['glaucoma', 'normal']:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                analysis['class_distribution'][class_name] = 0
                continue
                
            class_count = 0
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_count += 1
                    analysis['total_images'] += 1
                    
                    # Get file format
                    ext = filename.split('.')[-1].lower()
                    analysis['file_formats'][ext] = analysis['file_formats'].get(ext, 0) + 1
                    
                    # Get image size and check for corruption
                    image_path = os.path.join(class_dir, filename)
                    try:
                        with Image.open(image_path) as img:
                            analysis['image_sizes'].append(img.size)
                            # Verify image can be loaded
                            img.verify()
                    except Exception as e:
                        analysis['corrupted_images'] += 1
                        print(f"Corrupted image found: {image_path} - {e}")
            
            analysis['class_distribution'][class_name] = class_count
        
        return analysis
    
    def create_visualizations(self, data_dir, save_path=None):
        """Create dataset visualizations"""
        analysis = self.analyze_dataset(data_dir)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution pie chart
        if analysis['class_distribution']:
            axes[0, 0].pie(analysis['class_distribution'].values(), 
                          labels=analysis['class_distribution'].keys(), 
                          autopct='%1.1f%%')
            axes[0, 0].set_title('Class Distribution')
        
        # File format distribution
        if analysis['file_formats']:
            axes[0, 1].bar(analysis['file_formats'].keys(), analysis['file_formats'].values())
            axes[0, 1].set_title('File Format Distribution')
            axes[0, 1].set_xlabel('Format')
            axes[0, 1].set_ylabel('Count')
        
        # Image size distribution
        if analysis['image_sizes']:
            widths = [size[0] for size in analysis['image_sizes']]
            heights = [size[1] for size in analysis['image_sizes']]
            axes[1, 0].scatter(widths, heights, alpha=0.6)
            axes[1, 0].set_title('Image Size Distribution')
            axes[1, 0].set_xlabel('Width')
            axes[1, 0].set_ylabel('Height')
        
        # Sample images
        sample_images = []
        sample_labels = []
        for class_name in ['glaucoma', 'normal']:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    sample_path = os.path.join(class_dir, files[0])
                    img = self.load_and_preprocess_image(sample_path)
                    if img is not None:
                        sample_images.append(img)
                        sample_labels.append(class_name)
        
        if len(sample_images) >= 2:
            axes[1, 1].imshow(sample_images[0])
            axes[1, 1].set_title(f'Sample: {sample_labels[0]}')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return analysis
    
    def validate_dataset_integrity(self, data_dir):
        """Validate dataset integrity and report issues"""
        print(f"Validating dataset integrity for: {data_dir}")
        
        issues = []
        analysis = self.analyze_dataset(data_dir)
        
        # Check for corrupted images
        if analysis['corrupted_images'] > 0:
            issues.append(f"Found {analysis['corrupted_images']} corrupted images")
        
        # Check class balance
        class_counts = analysis['class_distribution']
        if len(class_counts) == 2:
            count_diff = abs(class_counts['glaucoma'] - class_counts['normal'])
            if count_diff > min(class_counts.values()) * 0.2:  # More than 20% difference
                issues.append(f"Class imbalance detected: {class_counts}")
        
        # Check for empty classes
        for class_name, count in class_counts.items():
            if count == 0:
                issues.append(f"Empty class: {class_name}")
        
        # Check file format consistency
        if len(analysis['file_formats']) > 2:
            issues.append(f"Multiple file formats detected: {analysis['file_formats']}")
        
        if issues:
            print("Dataset validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Dataset validation passed!")
        
        return issues, analysis 