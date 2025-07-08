"""
Dataset Manager for Computer Vision Classification

Handles multiple datasets with a unified interface:
- GTSRB (German Traffic Sign Recognition Benchmark)
- CIFAR-10 (Canadian Institute for Advanced Research)
- CIFAR-100 (Canadian Institute for Advanced Research - 100 classes)
- Fashion-MNIST
- MNIST
- Extensible for additional datasets
"""
from pathlib import Path
import os
import cv2
import numpy as np
import shutil
import importlib
from tensorflow.keras.datasets import cifar10 # type: ignore
from tensorflow.keras.datasets import cifar100 # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional, Callable
import tensorflow as tf
from utils.logger import logger, PerformanceLogger, TimedOperation
import kagglehub

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
        # This converts the labels to one-hot encoding wherein the possible labels (0=dog, 1=cat, 2=bird, etc.) are are converted to a vector in which all values are 0, except the position corresponding to the class label gets value 1.
        # This allows for the calculation of loss using categorical crossentropy (e.g. the prob output of [0.1, 0.05, 0.02, 0.8, 0.01, ...]  can now be subtracted from the one-hot labelling/prefect answer of [0.0, 0.0,  0.0,  1.0, 0.0,  ...] to calculate loss)
        # example: if there are 3 classes (labels):
        # label 0 becomes [1, 0, 0], 
        # label 1 becomes [0, 1, 0], and 
        # label 2 becomes [0, 0, 1]
        labels_categorical = to_categorical(labels, self.config.num_classes)
        
        # Split into train/test
        # train_test_split is a function from scikit-learn that randomly divides a dataset into training and testing portions.
        # 42 = random seed for reproducibility, e.g. if you run the code multiple times, you will get the same split of training and testing data.
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
        
        logger.debug(f"running class GTSRBLoader ... Loading GTSRB data from {data_dir}...")
        
        # Iterate through all categories (0-42)
        for category in range(self.config.num_classes):
            category_dir = os.path.join(data_dir, str(category))
            
            if not os.path.exists(category_dir):
                logger.warning(f"running class GTSRBLoader ... Warning: Category directory {category_dir} not found, skipping...")
                continue
            
            category_images = 0
            for file in os.listdir(category_dir):
                image_path = os.path.join(category_dir, file)
                
                # Read and preprocess image
                try:
                    # Open image using OpenCV
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Resize to target dimensions
                    image_resized = cv2.resize(image, (self.config.img_width, self.config.img_height))
                    
                    # Normalize pixel values to [0, 1]
                    # Rescales the pixel intensity values from the range [0, 255] to the range [0, 1]
                    # pixel = [255, 128, 0]  # Red=255 (max), Green=128 (medium), Blue=0 (none) --> normalized_pixel = [1.0, 0.5, 0.0]  # Red=1.0, Green=0.5, Blue=0.0
                    image_normalized = image_resized.astype('float32') / 255.0
                    
                    images.append(image_normalized)
                    labels.append(category)
                    category_images += 1
                    
                except Exception as e:
                    logger.error(f"running class GTSRBLoader ... Error processing {image_path}: {e}")
                    continue
            
            logger.debug(f"running class GTSRBLoader ... Category {category}: {category_images} images loaded")
        
        logger.debug(f"running class GTSRBLoader ... Total GTSRB images loaded: {len(images)}")
        return np.array(images), np.array(labels)

class KerasDatasetLoader(BaseDatasetLoader):
    """Generic loader for any Keras built-in dataset"""
    
    def __init__(self, config: DatasetConfig, dataset_module_path: str, class_names: Optional[List[str]] = None):
        """
        Initialize generic Keras dataset loader
        
        Args:
            config: Dataset configuration
            dataset_module_path: Module path like 'tensorflow.keras.datasets.cifar10'
            class_names: Optional manual class names, if None will try to auto-detect
        """
        super().__init__(config)
        self.dataset_module_path = dataset_module_path
        self.manual_class_names = class_names
        self._load_function: Optional[Callable[[], Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]] = None
        
    def _get_dataset_load_function(self) -> Callable[[], Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """Dynamically import and return the dataset's load_data function"""
        if self._load_function is None:
            try:
                # Import the dataset module dynamically
                # e.g., 'tensorflow.keras.datasets.cifar10' -> cifar10.load_data
                module = importlib.import_module(self.dataset_module_path)
                if not hasattr(module, 'load_data'):
                    raise ImportError(f"Module {self.dataset_module_path} does not have 'load_data' function")
                self._load_function = module.load_data
                logger.debug(f"running _get_dataset_load_function ... Successfully imported {self.dataset_module_path}")
            except ImportError as e:
                raise ImportError(f"Could not import {self.dataset_module_path}: {e}")
        
        # Type guard: at this point _load_function is guaranteed to not be None
        assert self._load_function is not None
        return self._load_function
    
    def _auto_detect_class_names(self) -> Optional[List[str]]:
        """
        Try to auto-detect class names from the dataset module
        
        Many Keras datasets have class_names as a module attribute
        """
        try:
            module = importlib.import_module(self.dataset_module_path)
            
            # Common attribute names for class names in Keras datasets
            possible_attrs = ['class_names', 'classes', 'labels', 'categories']
            
            for attr_name in possible_attrs:
                if hasattr(module, attr_name):
                    class_names = getattr(module, attr_name)
                    if isinstance(class_names, (list, tuple)) and len(class_names) == self.config.num_classes:
                        logger.debug(f"running _auto_detect_class_names ... Found class names via {attr_name}: {class_names}")
                        return list(class_names)
            
            logger.debug(f"running _auto_detect_class_names ... No class names found in {self.dataset_module_path}")
            return None
            
        except Exception as e:
            logger.warning(f"running _auto_detect_class_names ... Error auto-detecting class names: {e}")
            return None
    
    
    def load_data(self, data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from Keras built-in dataset"""
        logger.debug(f"running load_data ... Loading {self.config.name} from Keras...")
        
        # Get the load function
        load_function = self._get_dataset_load_function()
        
        # Determine if this is a text dataset and get appropriate parameters
        load_params = {}
        if (self.config.img_height == 1 and 
            self.config.channels == 1 and 
            self.config.img_width > 100):
            # This looks like text data - apply vocabulary control universally
            load_params = {
                'num_words': 10000,  # Standardize vocabulary across ALL text datasets
                'maxlen': self.config.img_width  # Use sequence length from config
            }
            logger.debug(f"running load_data ... Text dataset detected, applying vocab control: {load_params}")

        # Load the dataset
        try:
            if load_params:
                (x_train, y_train), (x_test, y_test) = load_function(**load_params)
            else:
                (x_train, y_train), (x_test, y_test) = load_function()
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.dataset_module_path}: {e}")
        
        # Combine train and test for our own splitting later
        images = np.concatenate([x_train, x_test])
        labels = np.concatenate([y_train, y_test])
        
        # Flatten labels if they come as [[0], [1], ...] instead of [0, 1, ...]
        if labels.ndim > 1 and labels.shape[1] == 1:
            labels = labels.flatten()
        
        # Check if this is text data (1D sequences) or image data (3D/4D arrays)
        if images.ndim == 1:
            logger.debug(f"running load_data ... Detected text data with {len(images)} sequences")
            
            # For text data, we need to pad sequences to uniform length
            from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
            
            # Use img_width as max_sequence_length for text datasets
            max_length = self.config.img_width
            
            # Pad sequences to uniform length
            images = pad_sequences(images, maxlen=max_length, padding='post', truncating='post')
            logger.debug(f"running load_data ... Padded sequences to length {max_length}")
            logger.debug(f"running load_data ... Final shape: {images.shape}")
            
            # For text, we don't normalize (sequences are integers, not pixel values)
            # We also don't resize or convert channels
            
        # Image data: proceed with image processing for 2-D and 3-D arrays, which represent images
        else:
            logger.debug(f"running load_data ... Detected image data with shape {images.shape}")
            
            # Resize images if needed
            current_shape = (images.shape[1], images.shape[2])
            target_shape = (self.config.img_width, self.config.img_height)
            
            if current_shape != target_shape:
                logger.debug(f"running load_data ... Resizing images from {current_shape} to {target_shape}")
                resized_images = []
                for img in images:
                    resized = cv2.resize(img, target_shape)
                    resized_images.append(resized)
                images = np.array(resized_images)
            
            # Handle grayscale images (convert to RGB if config expects 3 channels)
            if images.ndim == 3 and self.config.channels == 3:
                # Convert grayscale to RGB by repeating the channel
                images = np.stack([images] * 3, axis=-1)
                logger.debug(f"running load_data ... Converted grayscale to RGB")
            elif images.ndim == 4 and images.shape[-1] == 3 and self.config.channels == 1:
                # Convert RGB to grayscale
                images = np.dot(images[...,:3], [0.2989, 0.5870, 0.1140])
                images = np.expand_dims(images, axis=-1)
                logger.debug(f"running load_data ... Converted RGB to grayscale")
            
            # Normalize pixel values to [0, 1] for images only
            images = images.astype('float32') / 255.0
        
        # Update config with class names if available
        if self.config.class_names is None:
            detected_names = self.manual_class_names or self._auto_detect_class_names()
            if detected_names:
                self.config.class_names = detected_names
                logger.debug(f"running load_data ... Updated config with class names: {detected_names}")
            else:
                # Fallback to generic names
                self.config.class_names = [f"Class_{i}" for i in range(self.config.num_classes)]
                logger.debug(f"running load_data ... Using fallback class names: {self.config.class_names}")
        
        logger.debug(f"running load_data ... Total {self.config.name} samples loaded: {len(images)}")
        return images, labels

class DatasetManager:
    """Main manager class that handles multiple datasets with generic Keras support"""
    
    # Predefined dataset configurations - now includes CIFAR-100 and uses None for auto-detection
    DATASETS = {
        'gtsrb': DatasetConfig(
            name="German Traffic Signs (GTSRB)",
            num_classes=43,
            img_width=30,
            img_height=30,
            channels=3,
            folder_structure="folder_per_class",
            class_names=None  # Will be auto-generated as Class_0, Class_1, etc.
        ),
        'cifar10': DatasetConfig(
            name="CIFAR-10",
            num_classes=10,
            img_width=32,
            img_height=32,
            channels=3,
            folder_structure="builtin_keras",
            class_names=None  # Will be auto-detected from keras dataset
        ),
        'cifar100': DatasetConfig(
            name="CIFAR-100",
            num_classes=100,
            img_width=32,
            img_height=32,
            channels=3,
            folder_structure="builtin_keras",
            class_names=None  # Will be auto-detected from keras dataset
        ),
        'fashion_mnist': DatasetConfig(
            name="Fashion-MNIST",
            num_classes=10,
            img_width=28,
            img_height=28,
            channels=1,  # Grayscale
            folder_structure="builtin_keras",
            class_names=None  # Will be auto-detected
        ),
        'mnist': DatasetConfig(
            name="MNIST Digits",
            num_classes=10,
            img_width=28,
            img_height=28,
            channels=1,  # Grayscale
            folder_structure="builtin_keras",
            class_names=None  # Will be auto-detected or use fallback
        ),
        'imdb': DatasetConfig(
            name="IMDB Movie Reviews", 
            num_classes=2,
            img_width=500,  # Using img_width to represent max sequence length
            img_height=1,   # Not used for text, but required by current DatasetConfig
            channels=1,     # Not used for text, but required by current DatasetConfig
            folder_structure="builtin_keras",
            class_names=['Negative', 'Positive']  # 0=negative, 1=positive
        ),
        'reuters': DatasetConfig(
            name="Reuters Newswire Topics",
            num_classes=46,  # 46 news categories
            img_width=1000,  # Using img_width to represent max sequence length  
            img_height=1,    # Not used for text
            channels=1,      # Not used for text
            folder_structure="builtin_keras",
            class_names=None  # Will be auto-detected or use fallback
        )
    }
    
    # Keras dataset module mappings - maps dataset name to import path
    KERAS_DATASETS = {
        'cifar10': 'tensorflow.keras.datasets.cifar10', # https://keras.io/api/datasets/cifar10/
        'cifar100': 'tensorflow.keras.datasets.cifar100', # https://keras.io/api/datasets/cifar100/
        'fashion_mnist': 'tensorflow.keras.datasets.fashion_mnist', # https://keras.io/api/datasets/fashion_mnist/
        'mnist': 'tensorflow.keras.datasets.mnist', # https://keras.io/api/datasets/mnist/
        'imdb': 'tensorflow.keras.datasets.imdb', # https://keras.io/api/datasets/imdb/
        'reuters': 'tensorflow.keras.datasets.reuters' # https://keras.io/api/datasets/reuters/
    }
    
    # Dataset download configurations (only for non-Keras datasets)
    DOWNLOAD_CONFIGS = {
        'gtsrb': {
            'kaggle_dataset': 'meowmeowmeowmeowmeow/gtsrb-german-traffic-sign', # https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
            'fallback_url': 'https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip'
        }
    }
    
    def __init__(self, datasets_root: Optional[str] = None):
        """
        Initialize DatasetManager with optional custom datasets directory
        
        Args:
            datasets_root: Optional path to datasets directory. 
                          If None, uses project_root/datasets/
        """
        # Initialize loader mapping - now uses generic approach for Keras datasets
        self.loaders = {
            'gtsrb': GTSRBLoader,  # Keep custom loader for folder-based datasets
            # All Keras datasets will use the generic loader (initialized dynamically)
        }
        
        # Set up datasets directory path
        if datasets_root is None:
            from pathlib import Path
            current_file = Path(__file__)
            project_root = current_file.parent.parent  # Go up from src/ to project root
            self.datasets_root = project_root / "datasets"
        else:
            from pathlib import Path
            self.datasets_root = Path(datasets_root)
        
        # Create datasets directory if it doesn't exist
        self.datasets_root.mkdir(exist_ok=True)
        logger.debug(f"running class DatasetManager ... Datasets directory: {self.datasets_root}")
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names"""
        return list(self.DATASETS.keys())
    
    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get configuration for a specific dataset"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {self.get_available_datasets()}")
        return self.DATASETS[dataset_name]
    
    def _get_loader_for_dataset(self, dataset_name: str):
        """
        Get appropriate loader class for dataset (generic or custom)
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Loader class instance
        """
        config = self.get_dataset_config(dataset_name)
        
        # Check if it's a Keras dataset
        if dataset_name in self.KERAS_DATASETS:
            # Use generic Keras loader
            module_path = self.KERAS_DATASETS[dataset_name]
            
            # Try to get manual class names for specific datasets
            manual_class_names = self._get_manual_class_names(dataset_name)
            
            return KerasDatasetLoader(config, module_path, manual_class_names)
        
        # Use custom loader for non-Keras datasets
        elif dataset_name in self.loaders:
            loader_class = self.loaders[dataset_name]
            return loader_class(config)
        
        else:
            raise ValueError(f"No loader available for dataset '{dataset_name}'")
    
    def _get_manual_class_names(self, dataset_name: str) -> Optional[List[str]]:
        """
        Get manual class names for datasets where auto-detection might not work
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Manual class names if available, None otherwise
        """
        # Only add manual class names if auto-detection doesn't work well
        manual_names = {
            'fashion_mnist': [
                'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
            ],
            'mnist': [f'Digit_{i}' for i in range(10)]  # 0, 1, 2, ..., 9
        }
        
        return manual_names.get(dataset_name)
    
    def _download_progress_hook(self, block_num: int, block_size: int, total_size: int) -> None:
        """Show download progress"""
        downloaded = block_num * block_size
        progress = min(downloaded / total_size, 1.0) if total_size > 0 else 0
        percent = progress * 100
        
        # Show progress every 10%
        if block_num % max(1, int(total_size / block_size / 10)) == 0:
            logger.debug(f"running _download_progress_hook ... Download progress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)")
    
    def _download_and_extract_gtsrb(self, extract_to: Path) -> bool:
        """
        Download and extract GTSRB dataset from Kaggle (preferred) or fallback URL
        
        Args:
            extract_to: Directory to extract the dataset to
            
        Returns:
            True if successful, False otherwise
        """
        config = self.DOWNLOAD_CONFIGS['gtsrb']
        
        # Try Kaggle first (preferred method)
        try:
            logger.info(f"running _download_and_extract_gtsrb ... Downloading GTSRB from Kaggle dataset: {config['kaggle_dataset']}")
            logger.info(f"running _download_and_extract_gtsrb ... This may take a few minutes...")
            
            # Download using kagglehub
            kaggle_path = kagglehub.dataset_download(config['kaggle_dataset'])
            logger.debug(f"running _download_and_extract_gtsrb ... Kaggle downloaded to: {kaggle_path}")
            
            # Find the actual GTSRB data in the downloaded path
            kaggle_path = Path(kaggle_path)
            gtsrb_source = self._find_gtsrb_data_in_kaggle(kaggle_path)
            
            if gtsrb_source:
                # Copy/move the data to our expected location
                logger.info(f"running _download_and_extract_gtsrb ... Organizing dataset to {extract_to}")
                
                # Create target directory
                extract_to.mkdir(parents=True, exist_ok=True)
                
                # Copy the organized data
                if gtsrb_source != extract_to:
                    self._copy_gtsrb_structure(gtsrb_source, extract_to)
                
                # Verify the structure
                if self._validate_gtsrb_structure(extract_to):
                    logger.info(f"running _download_and_extract_gtsrb ... GTSRB dataset successfully downloaded from Kaggle to {extract_to}")
                    return True
                else:
                    logger.warning(f"running _download_and_extract_gtsrb ... Kaggle download completed but structure validation failed")
                    
            else:
                logger.warning(f"running _download_and_extract_gtsrb ... Could not find GTSRB data in Kaggle download")
                
        except Exception as e:
            logger.warning(f"running _download_and_extract_gtsrb ... Kaggle download failed: {e}")
            logger.info(f"running _download_and_extract_gtsrb ... Falling back to direct download...")
        
        # Fallback to direct download if Kaggle failed or unavailable
        return self._download_from_url_fallback(extract_to, config['fallback_url'])
    
    def _find_gtsrb_data_in_kaggle(self, kaggle_path: Path) -> Optional[Path]:
        """
        Find the actual GTSRB data structure in the Kaggle download
        
        Args:
            kaggle_path: Path where Kaggle downloaded the data
            
        Returns:
            Path to the directory containing 0/, 1/, 2/, ... folders, or None if not found
        """
        # Common patterns for GTSRB data organization in Kaggle datasets
        search_patterns = [
            kaggle_path,  # Data might be directly in the download path
            kaggle_path / "gtsrb",
            kaggle_path / "GTSRB", 
            kaggle_path / "German Traffic Sign Recognition Benchmark",
            kaggle_path / "data",
            kaggle_path / "train",
            kaggle_path / "training"
        ]
        
        # Also search recursively for any directory containing numbered folders
        for root, dirs, files in os.walk(kaggle_path):
            root_path = Path(root)
            # Check if this directory contains numbered subdirectories (0, 1, 2, etc.)
            numbered_dirs = [d for d in dirs if d.isdigit() and 0 <= int(d) <= 42]
            if len(numbered_dirs) >= 10:  # Require at least 10 numbered directories
                search_patterns.append(root_path)
        
        # Test each pattern
        for pattern in search_patterns:
            if pattern.exists() and self._validate_gtsrb_structure(pattern):
                logger.debug(f"running _find_gtsrb_data_in_kaggle ... Found GTSRB structure at: {pattern}")
                return pattern
        
        return None
    
    def _copy_gtsrb_structure(self, source: Path, target: Path) -> None:
        """
        Copy GTSRB directory structure from source to target
        
        Args:
            source: Source directory with GTSRB structure
            target: Target directory to copy to
        """
        logger.debug(f"running _copy_gtsrb_structure ... Copying from {source} to {target}")
        
        # Copy each numbered directory (0-42)
        for i in range(43):
            source_dir = source / str(i)
            target_dir = target / str(i)
            
            if source_dir.exists():
                if target_dir.exists():
                    shutil.rmtree(target_dir)  # Remove existing
                shutil.copytree(source_dir, target_dir)
                logger.debug(f"running _copy_gtsrb_structure ... Copied category {i}")
    
    def _download_from_url_fallback(self, extract_to: Path, url: str) -> bool:
        """
        Fallback method: download from direct URL (original CS50 method)
        
        Args:
            extract_to: Directory to extract to
            url: URL to download from
            
        Returns:
            True if successful
        """
        try:
            import urllib.request
            import zipfile
            
            zip_path = extract_to.parent / "gtsrb.zip"
            
            logger.info(f"running _download_from_url_fallback ... Downloading GTSRB from fallback URL: {url}")
            logger.info(f"running _download_from_url_fallback ... This may take a few minutes...")
            
            # Download the file
            urllib.request.urlretrieve(url, zip_path, self._download_progress_hook)
            
            logger.info(f"running _download_from_url_fallback ... Download completed. Extracting...")
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to.parent)
            
            # Clean up the zip file
            zip_path.unlink()
            
            # Verify extraction was successful
            if self._validate_gtsrb_structure(extract_to):
                logger.info(f"running _download_from_url_fallback ... GTSRB dataset successfully downloaded from URL to {extract_to}")
                return True
            else:
                logger.error(f"running _download_from_url_fallback ... Download completed but dataset structure is invalid")
                return False
                
        except Exception as e:
            logger.error(f"running _download_from_url_fallback ... Failed to download from URL: {e}")
            return False
    
    def _validate_gtsrb_structure(self, gtsrb_path: Path) -> bool:
        """
        Validate that GTSRB dataset has the expected structure
        
        Args:
            gtsrb_path: Path to the GTSRB dataset directory
            
        Returns:
            True if structure is valid
        """
        if not gtsrb_path.exists():
            return False
        
        # Check for expected directories (0-42)
        expected_dirs = set(str(i) for i in range(43))
        existing_dirs = set(d.name for d in gtsrb_path.iterdir() if d.is_dir() and d.name.isdigit())
        
        # Require at least 40 out of 43 categories (some flexibility for missing categories)
        common_dirs = expected_dirs & existing_dirs
        if len(common_dirs) >= 40:
            logger.debug(f"running _validate_gtsrb_structure ... GTSRB structure validation passed: {len(common_dirs)}/43 categories found")
            return True
        else:
            logger.warning(f"running _validate_gtsrb_structure ... GTSRB structure validation failed: only {len(common_dirs)}/43 categories found")
            return False
    
    def load_dataset(self, dataset_name: str, data_dir: Optional[str] = None, test_size: float = 0.4, auto_download: bool = True) -> Dict[str, Any]:
        """
        Load and prepare a dataset for training with automatic download support
        Enhanced to work with generic Keras datasets
        
        Args:
            dataset_name: Name of the dataset to load
            data_dir: Optional explicit data directory. If None, uses automatic path resolution
            test_size: Fraction of data to use for testing
            auto_download: Whether to automatically download missing datasets
            
        Returns:
            Dictionary containing training/test data and configuration
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {self.get_available_datasets()}")
        
        config = self.DATASETS[dataset_name]
        
        # Get appropriate loader
        loader = self._get_loader_for_dataset(dataset_name)
        
        # Handle data directory resolution for folder-based datasets only
        if config.folder_structure == "folder_per_class":
            if data_dir is None:
                data_dir = str(self.datasets_root / dataset_name)
                
                # Check if dataset exists
                dataset_path = Path(data_dir)
                if not dataset_path.exists() or not self._validate_dataset_structure(dataset_name, dataset_path):
                    if auto_download and dataset_name in self.DOWNLOAD_CONFIGS:
                        logger.info(f"running load_dataset ... {config.name} not found locally. Attempting automatic download...")
                        
                        if dataset_name == 'gtsrb':
                            success = self._download_and_extract_gtsrb(dataset_path)
                            if not success:
                                self._raise_dataset_not_found_error(dataset_name, data_dir)
                        else:
                            logger.error(f"running load_dataset ... Automatic download not implemented for {dataset_name}")
                            self._raise_dataset_not_found_error(dataset_name, data_dir)
                    else:
                        self._raise_dataset_not_found_error(dataset_name, data_dir)
        else:
            # For Keras datasets, data_dir is not used
            data_dir = None
        
        logger.debug(f"running load_dataset ... Loading {config.name} from {data_dir or 'built-in Keras source'}")
        return loader.prepare_data(data_dir, test_size)
    
    def _validate_dataset_structure(self, dataset_name: str, dataset_path: Path) -> bool:
        """
        Validate dataset structure based on dataset type
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to the dataset directory
            
        Returns:
            True if structure is valid
        """
        if dataset_name == 'gtsrb':
            return self._validate_gtsrb_structure(dataset_path)
        else:
            # For other datasets, just check if directory exists
            return dataset_path.exists()
    
    def _raise_dataset_not_found_error(self, dataset_name: str, data_dir: str):
        """Raise helpful error when dataset is not found"""
        if dataset_name == 'gtsrb':
            error_msg = f"""
GTSRB dataset not found at {data_dir}

Automatic download failed. You can try:
1. Install kagglehub: pip install kagglehub
2. Set up Kaggle API credentials (if needed)
3. Or manually download: https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip
4. Extract to: {data_dir}
5. Expected structure: {data_dir}/0/, {data_dir}/1/, ..., {data_dir}/42/

Or try running again with a stable internet connection.
"""
        else:
            error_msg = f"Dataset '{dataset_name}' not found at {data_dir}"
        
        logger.error(f"running class DatasetManager ... {error_msg}")
        raise FileNotFoundError(error_msg)
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """
        Manually download a dataset
        
        Args:
            dataset_name: Name of the dataset to download
            force: Whether to re-download if dataset already exists
            
        Returns:
            True if successful
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {self.get_available_datasets()}")
        
        if dataset_name not in self.DOWNLOAD_CONFIGS:
            logger.error(f"running download_dataset ... No download configuration available for {dataset_name}")
            return False
        
        dataset_path = self.datasets_root / dataset_name
        
        if dataset_path.exists() and not force:
            if self._validate_dataset_structure(dataset_name, dataset_path):
                logger.info(f"running download_dataset ... {dataset_name} already exists and is valid")
                return True
            else:
                logger.info(f"running download_dataset ... {dataset_name} exists but is invalid, re-downloading...")
        
        if dataset_name == 'gtsrb':
            return self._download_and_extract_gtsrb(dataset_path)
        else:
            logger.error(f"running download_dataset ... Download not implemented for {dataset_name}")
            return False
    
    def add_custom_dataset(self, name: str, config: DatasetConfig, loader_class):
        """Add a custom dataset configuration for folder-based datasets"""
        self.DATASETS[name] = config
        self.loaders[name] = loader_class
        logger.debug(f"running class DatasetManager, add_custom_dataset ... Added custom dataset: {name}")
    
    def add_keras_dataset(self, name: str, config: DatasetConfig, module_path: str, class_names: Optional[List[str]] = None) -> None:
        """
        Add a custom Keras dataset configuration
        
        Args:
            name: Dataset name
            config: Dataset configuration
            module_path: Import path like 'tensorflow.keras.datasets.custom_dataset'
            class_names: Optional manual class names
        """
        self.DATASETS[name] = config
        self.KERAS_DATASETS[name] = module_path
        logger.debug(f"running add_keras_dataset ... Added Keras dataset: {name} from {module_path}")

# Example usage and testing
if __name__ == "__main__":
    manager = DatasetManager()
    
    logger.debug("running dataset_manager.py ... Available datasets:")
    for dataset in manager.get_available_datasets():
        config = manager.get_dataset_config(dataset)
        logger.debug(f"- {dataset}: {config.name} ({config.num_classes} classes, {config.input_shape})")
    
    # Test CIFAR-10 loading
    logger.debug("\nTesting CIFAR-10 loading...")
    try:
        data = manager.load_dataset('cifar10')
        logger.debug(f"CIFAR-10 loaded successfully!")
        logger.debug(f"Training set: {data['x_train'].shape}")
        logger.debug(f"Test set: {data['x_test'].shape}")
        logger.debug(f"Class names: {data['config'].class_names}")
    except Exception as e:
        logger.error(f"Error loading CIFAR-10: {e}")
    
    # Test CIFAR-100 loading (new!)
    logger.debug("\nTesting CIFAR-100 loading...")
    try:
        data = manager.load_dataset('cifar100', test_size=0.2)
        logger.debug(f"CIFAR-100 loaded successfully!")
        logger.debug(f"Training set: {data['x_train'].shape}")
        logger.debug(f"Test set: {data['x_test'].shape}")
        logger.debug(f"Number of classes: {len(data['config'].class_names) if data['config'].class_names else 'None'}")
        if data['config'].class_names:
            logger.debug(f"First 10 class names: {data['config'].class_names[:10]}")
    except Exception as e:
        logger.error(f"Error loading CIFAR-100: {e}")
        
    # Test Fashion-MNIST loading (new!)
    logger.debug("\nTesting Fashion-MNIST loading...")
    try:
        data = manager.load_dataset('fashion_mnist', test_size=0.2)
        logger.debug(f"Fashion-MNIST loaded successfully!")
        logger.debug(f"Training set: {data['x_train'].shape}")
        logger.debug(f"Test set: {data['x_test'].shape}")
        logger.debug(f"Class names: {data['config'].class_names}")
    except Exception as e:
        logger.error(f"Error loading Fashion-MNIST: {e}")