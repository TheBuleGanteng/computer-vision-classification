"""
Dataset Manager for Computer Vision Classification

Handles multiple datasets with a unified interface:
- GTSRB (German Traffic Sign Recognition Benchmark)
- CIFAR-10 (Canadian Institute for Advanced Research)
- Extensible for additional datasets
"""

import os
import cv2
import numpy as np
from tensorflow.keras.datasets import cifar10 # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import tensorflow as tf

@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str
    num_classes: int
    img_width: int
    img_height: int
    channels: int
    folder_structure: str  # "folder_per_class" or "builtin_keras"
    class_names: Optional[List[str]] = None
    
    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.img_width, self.img_height, self.channels)

class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    @abstractmethod
    def load_data(self, data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw images and labels"""
        pass
    
    def prepare_data(self, data_dir: Optional[str] = None, test_size: float = 0.4) -> Dict[str, Any]:
        """Load, preprocess, and split data into train/test sets"""
        images, labels = self.load_data(data_dir)
        
        # Convert labels to categorical
        labels_categorical = to_categorical(labels, self.config.num_classes)
        
        # Split into train/test
        x_train, x_test, y_train, y_test = train_test_split(
            images, labels_categorical, test_size=test_size, random_state=42
        )
        
        return {
            'x_train': x_train,
            'x_test': x_test, 
            'y_train': y_train,
            'y_test': y_test,
            'config': self.config
        }

class GTSRBLoader(BaseDatasetLoader):
    """Loader for German Traffic Sign Recognition Benchmark"""
    
    def load_data(self, data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load GTSRB data from directory structure"""
        if not data_dir or not os.path.exists(data_dir):
            raise ValueError(f"GTSRB data directory is required and must exist: {data_dir}")
        
        images = []
        labels = []
        
        print(f"Loading GTSRB data from {data_dir}...")
        
        # Iterate through all categories (0-42)
        for category in range(self.config.num_classes):
            category_dir = os.path.join(data_dir, str(category))
            
            if not os.path.exists(category_dir):
                print(f"Warning: Category directory {category_dir} not found, skipping...")
                continue
            
            category_images = 0
            for file in os.listdir(category_dir):
                image_path = os.path.join(category_dir, file)
                
                try:
                    # Read and preprocess image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Resize to target dimensions
                    image_resized = cv2.resize(image, (self.config.img_width, self.config.img_height))
                    
                    # Normalize pixel values to [0, 1]
                    image_normalized = image_resized.astype('float32') / 255.0
                    
                    images.append(image_normalized)
                    labels.append(category)
                    category_images += 1
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
            
            print(f"Category {category}: {category_images} images loaded")
        
        print(f"Total GTSRB images loaded: {len(images)}")
        return np.array(images), np.array(labels)

class CIFAR10Loader(BaseDatasetLoader):
    """Loader for CIFAR-10 dataset (built into Keras)"""
    
    def load_data(self, data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load CIFAR-10 data from Keras datasets"""
        print("Loading CIFAR-10 data from Keras...")
        
        # Download/load CIFAR-10 (happens automatically)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Combine train and test for our own splitting later
        images = np.concatenate([x_train, x_test])
        labels = np.concatenate([y_train, y_test])
        
        # Flatten labels (they come as [[0], [1], ...] instead of [0, 1, ...])
        labels = labels.flatten()
        
        # Resize if needed (CIFAR-10 is 32x32, might want different size)
        if (images.shape[1], images.shape[2]) != (self.config.img_width, self.config.img_height):
            print(f"Resizing CIFAR-10 from 32x32 to {self.config.img_width}x{self.config.img_height}")
            resized_images = []
            for img in images:
                resized = cv2.resize(img, (self.config.img_width, self.config.img_height))
                resized_images.append(resized)
            images = np.array(resized_images)
        
        # Normalize pixel values to [0, 1]
        images = images.astype('float32') / 255.0
        
        print(f"Total CIFAR-10 images loaded: {len(images)}")
        return images, labels

class DatasetManager:
    """Main manager class that handles multiple datasets"""
    
    # Predefined dataset configurations
    DATASETS = {
        'gtsrb': DatasetConfig(
            name="German Traffic Signs (GTSRB)",
            num_classes=43,
            img_width=30,
            img_height=30,
            channels=3,
            folder_structure="folder_per_class",
            class_names=[f"Class_{i}" for i in range(43)]  # Simplified names
        ),
        'cifar10': DatasetConfig(
            name="CIFAR-10",
            num_classes=10,
            img_width=32,
            img_height=32,
            channels=3,
            folder_structure="builtin_keras",
            class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck']
        )
    }
    
    def __init__(self):
        self.loaders = {
            'gtsrb': GTSRBLoader,
            'cifar10': CIFAR10Loader
        }
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names"""
        return list(self.DATASETS.keys())
    
    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get configuration for a specific dataset"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {self.get_available_datasets()}")
        return self.DATASETS[dataset_name]
    
    def load_dataset(self, dataset_name: str, data_dir: Optional[str] = None, test_size: float = 0.4) -> Dict[str, Any]:
        """Load and prepare a dataset for training"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {self.get_available_datasets()}")
        
        config = self.DATASETS[dataset_name]
        loader_class = self.loaders[dataset_name]
        loader = loader_class(config)
        
        return loader.prepare_data(data_dir, test_size)
    
    def add_custom_dataset(self, name: str, config: DatasetConfig, loader_class):
        """Add a custom dataset configuration"""
        self.DATASETS[name] = config
        self.loaders[name] = loader_class
        print(f"Added custom dataset: {name}")

# Example usage and testing
if __name__ == "__main__":
    manager = DatasetManager()
    
    print("Available datasets:")
    for dataset in manager.get_available_datasets():
        config = manager.get_dataset_config(dataset)
        print(f"- {dataset}: {config.name} ({config.num_classes} classes, {config.input_shape})")
    
    # Test CIFAR-10 loading
    print("\nTesting CIFAR-10 loading...")
    try:
        data = manager.load_dataset('cifar10')
        print(f"CIFAR-10 loaded successfully!")
        print(f"Training set: {data['x_train'].shape}")
        print(f"Test set: {data['x_test'].shape}")
        print(f"Class names: {data['config'].class_names}")
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")