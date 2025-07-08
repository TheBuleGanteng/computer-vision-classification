# Computer Vision Classification Project

This project implements a comprehensive **multi-modal machine learning system** for classification tasks using convolutional neural networks (CNNs) for images and LSTM networks for text. The system features a modular, extensible architecture with automatic dataset management, modern CNN optimizations, and comprehensive logging capabilities.

## 🎯 Project Objectives

### Core Goals
1. **Multi-Modal Classification Support**: Unified interface for both computer vision (CNN) and natural language processing (LSTM) tasks ✅
2. **Modern CNN Architecture**: Implement efficient architectures with global pooling options for parameter reduction ✅
3. **Automated Dataset Management**: Support multiple datasets with automatic downloading and preprocessing ✅
4. **Hyperparameter Optimization**: Bayesian optimization for automated model tuning 🚧
5. **Web Interface & API**: FastAPI backend with modern frontend for model configuration and training 🚧
6. **Interactive Visualizations**: Animated model architecture diagrams showing information flow and performance 🚧

### Extended Roadmap
1. **Enhanced Testing & Logging**: Detailed per-prediction analysis with visual feedback ⏳
2. **Universal Optimizer**: Generic hyperparameter optimization for all supported datasets ⏳
3. **API Development**: FastAPI endpoints for model training, evaluation, and download ⏳
4. **Frontend Interface**: React/Vue.js interface for dataset selection and hyperparameter configuration ⏳
5. **Architecture Visualization**: Interactive 3D animations showing CNN/LSTM architectures and training flow ⏳

## 🏗️ Current Architecture

### 1. Multi-Modal Dataset Management (`dataset_manager.py`)

**Supported Datasets:**

**Image Classification:**
- **GTSRB**: German Traffic Signs (43 classes, 30×30×3, folder-based with auto-download)
- **CIFAR-10**: 10 object classes (32×32×3, built-in Keras)
- **CIFAR-100**: 100 object classes (32×32×3, built-in Keras)
- **Fashion-MNIST**: Fashion items (28×28×1, built-in Keras)
- **MNIST**: Handwritten digits (28×28×1, built-in Keras)

**Text Classification:**
- **IMDB**: Movie review sentiment (2 classes, 500 sequence length)
- **Reuters**: Newswire topics (46 classes, 1000 sequence length)

**Key Features:**
- **Automatic data type detection** (image vs text based on dimensions)
- **Multi-modal preprocessing pipeline** (image normalization, text tokenization/padding)
- **Kaggle integration** with fallback URLs for dataset downloading
- **Comprehensive validation** and error handling

### 2. Intelligent Model Builder (`model_builder.py`)

**Architecture Auto-Selection:**
```python
# Automatic detection based on data characteristics
if img_height == 1 and channels == 1 and img_width > 100:
    # Text data → Build LSTM architecture
else:
    # Image data → Build CNN architecture
```

**CNN Architecture (Images):**
```
Input Layer (dataset-specific dimensions)
    ↓
Convolutional Layers (Feature Detection)
├── Conv2D (configurable filters/kernels)
├── MaxPooling2D (dimensionality reduction)
└── [Repeated based on num_layers_conv]
    ↓
Pooling Strategy (CONFIGURABLE)
├── Traditional: Flatten (2D → 1D conversion)
└── Modern: GlobalAveragePooling2D (99% parameter reduction)
    ↓
Dense Layers (Classification)
├── Dense (configurable neurons with funnel effect)
├── Dropout (overfitting prevention)
└── [Repeated with decreasing neurons]
    ↓
Output Layer (softmax, num_classes neurons)
```

**LSTM Architecture (Text):**
```
Input Layer (sequence_length,)
    ↓
Embedding Layer (word index → dense vectors)
    ↓
Sequential Processing
├── LSTM (configurable units, dropout)
└── Optional: Bidirectional LSTM (forward + backward)
    ↓
Dense Layers (Feature Combination)
├── Dense (configurable neurons)
├── Dropout (regularization)
    ↓
Output Layer (softmax/sigmoid, num_classes neurons)
```

### 3. Performance Monitoring & Logging

**Comprehensive Logging System:**
- **Multi-modal operation timing** (model building, training, evaluation)
- **Data pipeline monitoring** (preprocessing steps, sequence processing)
- **Detailed architecture documentation** with layer-by-layer explanations
- **Performance metrics tracking** (accuracy, loss, parameter counts)

## 🚀 Quick Start Guide

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd computer-vision-classification

# Install dependencies
pip install -r requirements.txt
```

### Command Line Usage

**Basic Image Classification:**
```bash
# Train CIFAR-10 with modern architecture
python src/model_builder.py dataset_name=cifar10 use_global_pooling=true epochs=15

# Train Fashion-MNIST with custom configuration
python src/model_builder.py dataset_name=fashion_mnist epochs=20 filters_per_conv_layer=64
```

**Text Classification:**
```bash
# Train IMDB sentiment analysis
python src/model_builder.py dataset_name=imdb epochs=10 embedding_dim=256

# Train Reuters topic classification
python src/model_builder.py dataset_name=reuters lstm_units=128 use_bidirectional=true
```

**Load and Evaluate Existing Models:**
```bash
# Load saved model and evaluate on new dataset
python src/model_builder.py load_model=./saved_models/model_20250708_142614_LSTM_acc_76_0.keras dataset_name=imdb test_size=0.1

# Load CNN model and test on different dataset
python src/model_builder.py load_model=./saved_models/model_20250708_143022_CNN_acc_94_2.keras dataset_name=cifar100
```

**Architecture Comparison:**
```bash
# Compare traditional vs modern CNN architectures
python src/model_builder.py dataset_name=cifar10 use_global_pooling=false epochs=10
python src/model_builder.py dataset_name=cifar10 use_global_pooling=true epochs=10
```

### Configuration Parameters

**CNN Parameters:**
- `use_global_pooling`: Modern vs traditional architecture (true/false)
- `num_layers_conv`: Number of convolutional layers (1-4)
- `filters_per_conv_layer`: Filters per layer (16, 32, 64, 128, 256)
- `kernel_size`: Filter size (3,3 or 5,5)
- `pool_size`: Pooling size (2,2 or 3,3)

**LSTM Parameters:**
- `embedding_dim`: Word embedding dimensions (64-512)
- `lstm_units`: LSTM memory cells (32-256)
- `use_bidirectional`: Bidirectional processing (true/false)
- `text_dropout`: Text-specific dropout rate (0.1-0.7)

**Training Parameters:**
- `epochs`: Training iterations (5-50)
- `num_layers_hidden`: Dense layers (1-5)
- `first_hidden_layer_nodes`: Hidden layer neurons (64-512)
- `test_size`: Test data fraction (0.1-0.4)

## 📊 Current Performance Results

### Multi-Modal Performance
**Image Classification:**
- **CIFAR-10**: 85-90% accuracy (traditional: 1.25M params, modern: 850K params)
- **Fashion-MNIST**: 90-93% accuracy
- **MNIST**: 98%+ accuracy  
- **GTSRB**: 96%+ accuracy

**Text Classification:**
- **IMDB**: 85-88% accuracy (sentiment analysis)
- **Reuters**: 76-85% accuracy (topic classification)

### Architecture Efficiency Gains
**GlobalAveragePooling vs Flatten:**
```python
# Traditional (parameter-heavy):
# (6×6×32 = 1152) × 128 = 147,456 parameters

# Modern (efficient):  
# 32 → 43 = 1,376 parameters (99% reduction!)
```

**Real Performance Example (CIFAR-10):**
- Traditional CNN: 85.2% accuracy, 1,250,000 parameters
- Modern CNN: 84.8% accuracy, 850,000 parameters (32% reduction, minimal accuracy loss)

## 🔧 Development Progress

### ✅ Completed Features

**Core Infrastructure:**
- [x] Multi-modal dataset support (images + text)
- [x] Automatic architecture selection based on data type
- [x] Modern CNN optimizations (GlobalAveragePooling)
- [x] Comprehensive logging and performance monitoring
- [x] Model saving/loading with intelligent naming
- [x] Command-line interface with flexible parameter parsing

**Dataset Integration:**
- [x] GTSRB with Kaggle auto-download
- [x] All major Keras datasets (CIFAR-10/100, Fashion-MNIST, MNIST)
- [x] Text datasets (IMDB, Reuters) with preprocessing
- [x] Unified preprocessing pipeline for both modalities

**Model Architecture:**
- [x] CNN architecture with configurable layers
- [x] LSTM architecture with bidirectional support
- [x] Automatic text vs image detection
- [x] Dropout and regularization strategies
- [x] Parameter efficiency optimizations

### 🚧 In Progress

**1. Enhanced Testing & Logging**
- [ ] Per-prediction analysis with visual feedback
- [ ] Confusion matrix generation and visualization
- [ ] Detailed error analysis (✅ Correct, predicted: cat, actual: dog)
- [ ] Training progress visualization

**2. Universal Hyperparameter Optimization**
- [ ] Generic optimizer for all datasets (extending traffic_optimize.py)
- [ ] Multi-modal search spaces (CNN + LSTM parameters)
- [ ] Bayesian optimization with Optuna integration
- [ ] Automated model comparison and selection

### ⏳ Next Development Phase

**3. API Development (FastAPI)**
```python
# Planned endpoints:
POST /api/train          # Start training with configuration
GET  /api/datasets       # List available datasets
POST /api/optimize       # Run hyperparameter optimization
GET  /api/models         # List saved models
POST /api/evaluate       # Evaluate model on test data
GET  /api/download/{id}  # Download trained model
```

**4. Frontend Interface**
- [ ] React/Vue.js interface for dataset selection
- [ ] Interactive hyperparameter configuration
- [ ] Real-time training progress monitoring
- [ ] Model download and export functionality
- [ ] Performance comparison dashboard

**5. Interactive Architecture Visualization**
- [ ] 3D animated model architecture diagrams
- [ ] Forward/backward pass visualization
- [ ] Filter activation maps for CNNs
- [ ] Attention visualization for LSTMs
- [ ] Performance metrics overlay

## 🛠️ Technical Implementation Details

### Modern CNN Optimization Results
```python
# Parameter Reduction Example:
Traditional Architecture:
Conv → Pool → Conv → Pool → Flatten → Dense(128) → Output
# Final conv output: (6, 6, 32) = 1,152 features
# Dense layer: 1,152 × 128 = 147,456 parameters

Modern Architecture:  
Conv → Pool → Conv → Pool → GlobalAveragePooling2D → Output
# GlobalAveragePooling: 32 features (one per filter)
# Output layer: 32 × 10 = 320 parameters (99.8% reduction!)
```

### Multi-Modal Data Pipeline
```python
# Automatic detection and processing:
def _detect_data_type(dataset_config):
    if (img_height == 1 and channels == 1 and img_width > 100):
        return "text"    # → Build LSTM
    else:
        return "image"   # → Build CNN

# Pipeline automatically handles:
# - Image: resize, normalize, RGB/grayscale conversion
# - Text: tokenize, pad sequences, vocabulary control
```

### File Structure
```
computer-vision-classification/
├── src/
│   ├── dataset_manager.py     # Multi-modal dataset handling
│   ├── model_builder.py       # CNN/LSTM architecture builder
│   ├── optimizer.py           # Hyperparameter optimization (planned)
│   └── utils/
│       └── logger.py          # Performance monitoring
├── datasets/                  # Auto-downloaded datasets
├── saved_models/             # Trained model storage
├── logs/                     # Training and performance logs
├── requirements.txt          # Dependencies
└── README.md                # This documentation
```

## 🎯 Upcoming Features

### Enhanced User Experience
1. **Web Interface**: Modern UI for model configuration and training
2. **Real-time Monitoring**: Live training progress and metrics
3. **Model Marketplace**: Save, share, and download optimized models
4. **Interactive Tutorials**: Guided learning for CNN/LSTM concepts

### Advanced ML Features
1. **Transfer Learning**: Pre-trained model integration
2. **Ensemble Methods**: Multi-model prediction combining
3. **AutoML**: Fully automated architecture search
4. **Production Deployment**: Docker containers and cloud integration

### Visualization & Education
1. **Architecture Animations**: 3D models showing information flow
2. **Filter Visualization**: CNN feature map exploration
3. **Training Dynamics**: Loss landscape and convergence visualization
4. **Educational Content**: Interactive ML concept explanations

## 🚀 Getting Started for New Users

This project is designed to be both a **production-ready ML system** and an **educational platform** for understanding modern deep learning architectures. Whether you're training models for research, comparing architectures, or learning about CNNs and LSTMs, the system provides comprehensive tools and clear documentation.

**Start with the quickstart commands above**, then explore the extensive configuration options and logging outputs to understand how modern neural networks achieve efficient, high-performance classification across multiple data modalities.

The upcoming web interface will make these capabilities accessible to users without command-line experience, while the architecture visualizations will provide intuitive understanding of how these models process information and make predictions.