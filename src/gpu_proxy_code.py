"""
GPU Proxy Training Code - MANUAL VALIDATION SPLIT FIX

This module contains the training code that gets executed on the GPU proxy environment.
Extracted from model_builder.py for better code organization and maintainability.

Key Features:
- Manual validation split implementation to bypass GPU validation_split bug
- Comprehensive environment diagnostics
- Enhanced validation metric analysis
- Phase 4 manual validation split fix implementation
"""
import sys
import traceback
import time

def get_gpu_proxy_training_code(validation_split_value: float) -> str:
    """
    Generate the complete training code for GPU proxy execution.
    
    Args:
        validation_split_value: The validation split ratio to use
        
    Returns:
        Complete Python code string for GPU proxy execution
    """
    
    training_code = f"""
import sys
import traceback
import time
print("=== STARTING REMOTE EXECUTION WITH MANUAL VALIDATION SPLIT FIX ===")
print(f"Python version: {{sys.version}}")

# FIXED: Initialize keras variables at the top level to avoid UnboundLocalError
keras_standalone = False
keras_version = None
keras_available = False

try:
    import tensorflow as tf
    keras_available = True
    print(f"TensorFlow version: {{tf.__version__}}")
    
    # CRITICAL: Comprehensive TensorFlow/Keras environment analysis
    print("=== COMPREHENSIVE TENSORFLOW/KERAS ENVIRONMENT ANALYSIS ===")
    print(f"TensorFlow version: {{tf.__version__}}")
    print(f"TensorFlow built with CUDA: {{tf.test.is_built_with_cuda()}}")
    print(f"TensorFlow CUDA version: {{tf.sysconfig.get_build_info().get('cuda_version', 'Unknown')}}")
    print(f"TensorFlow cuDNN version: {{tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown')}}")
    
    # Keras version analysis - FIXED: Proper variable scoping
    try:
        import keras
        keras_version = keras.__version__
        print(f"Standalone Keras version: {{keras_version}}")
        print(f"Standalone Keras location: {{keras.__file__}}")
        keras_standalone = True
    except ImportError:
        print("Standalone Keras: NOT AVAILABLE")
        keras_standalone = False
    
    # TensorFlow's built-in Keras
    print(f"TensorFlow Keras version: {{tf.keras.__version__}}")
    print(f"TensorFlow Keras location: {{tf.keras.__file__}}")
    
    # Version comparison
    keras_version = tf.keras.__version__
    if keras_standalone:
        print(f"Keras version match: {{keras_version == keras_version}}")
        print(f"TF Keras: {{keras_version}} vs Standalone: {{keras_version}}")
    else:
        print(f"Only TensorFlow Keras available: {{keras_version}}")
    
    # GPU and CUDA details
    print(f"Physical GPU devices: {{tf.config.list_physical_devices('GPU')}}")
    print(f"Logical GPU devices: {{tf.config.list_logical_devices('GPU')}}")
    
    # Build info
    build_info = tf.sysconfig.get_build_info()
    print(f"TensorFlow build info keys: {{list(build_info.keys())}}")
    for key, value in build_info.items():
        print(f"  {{key}}: {{value}}")
    
    # 🎯 PHASE 4: MANUAL VALIDATION SPLIT TESTING
    print("=== PHASE 4: MANUAL VALIDATION SPLIT IMPLEMENTATION ===")
    
    # Create a simple test model for validation_split comparison
    test_input_shape = (32, 32, 3)  # Sample shape
    test_num_classes = 10
    
    print("Testing BROKEN validation_split vs FIXED manual split:")
    test_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=test_input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(test_num_classes, activation='softmax')
    ])
    test_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    # Create minimal test data
    import numpy as np
    test_x = np.random.random((100, 32, 32, 3)).astype(np.float32)
    test_y = tf.keras.utils.to_categorical(np.random.randint(0, 10, 100), 10)
    
    validation_split_val = {validation_split_value}
    print(f"Testing with validation_split = {{validation_split_val}}")
    
    # TEST 1: Broken validation_split approach
    print("\\n--- TEST 1: BROKEN validation_split approach ---")
    broken_model = tf.keras.models.clone_model(test_model)
    broken_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    print("Training with validation_split parameter (BROKEN)...")
    broken_history = broken_model.fit(
        test_x, test_y,
        epochs=2,
        validation_split=validation_split_val,
        verbose=1
    )
    
    print(f"BROKEN approach results:")
    for key, values in broken_history.history.items():
        print(f"  {{key}}: {{values}}")
    
    # TEST 2: Fixed manual validation split approach
    print("\\n--- TEST 2: FIXED manual validation split approach ---")
    fixed_model = tf.keras.models.clone_model(test_model)
    fixed_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    # 🎯 MANUAL VALIDATION SPLIT IMPLEMENTATION
    print("Implementing manual validation split...")
    split_idx = int(len(test_x) * (1 - validation_split_val))
    
    x_train_split = test_x[:split_idx]
    y_train_split = test_y[:split_idx]
    x_val_manual = test_x[split_idx:]
    y_val_manual = test_y[split_idx:]
    
    print(f"Manual split: train={{len(x_train_split)}}, val={{len(x_val_manual)}}")
    print(f"Train data: x={{x_train_split.shape}}, y={{y_train_split.shape}}")
    print(f"Val data: x={{x_val_manual.shape}}, y={{y_val_manual.shape}}")
    
    print("Training with validation_data parameter (FIXED)...")
    fixed_history = fixed_model.fit(
        x_train_split, y_train_split,
        epochs=2,
        validation_data=(x_val_manual, y_val_manual),
        verbose=1
    )
    
    print(f"FIXED approach results:")
    for key, values in fixed_history.history.items():
        print(f"  {{key}}: {{values}}")
    
    # COMPARISON: Analyze the difference
    print("\\n=== VALIDATION SPLIT COMPARISON ANALYSIS ===")
    broken_val_acc = broken_history.history.get('val_categorical_accuracy', [])
    fixed_val_acc = fixed_history.history.get('val_categorical_accuracy', [])
    
    print(f"BROKEN val_categorical_accuracy: {{broken_val_acc}}")
    print(f"FIXED val_categorical_accuracy: {{fixed_val_acc}}")
    
    broken_working = len(broken_val_acc) > 0 and any(v > 0 for v in broken_val_acc)
    fixed_working = len(fixed_val_acc) > 0 and any(v > 0 for v in fixed_val_acc)
    
    print(f"BROKEN validation_split working: {{broken_working}}")
    print(f"FIXED manual split working: {{fixed_working}}")
    
    if fixed_working and not broken_working:
        print("✅ MANUAL VALIDATION SPLIT FIX CONFIRMED!")
        print("✅ Manual approach bypasses the GPU validation_split bug")
    elif broken_working and fixed_working:
        print("⚠️  Both approaches working - may be environment dependent")
    else:
        print("❌ Both approaches failing - deeper investigation needed")
    
    # FIXED: Use tf.keras consistently to avoid import conflicts - use alias to prevent overwriting
    from tensorflow import keras
    import numpy as np
    import json
    import gzip
    import base64
    
    print("=== IMPORTS SUCCESSFUL ===")
    
    # Record training start time
    training_start_time = time.time()
    
    # Debug context data
    print(f"Context keys: {{list(context.keys())}}")
    print(f"Compressed data: {{context.get('compressed', False)}}")
    
    # Handle compressed vs uncompressed data
    if context.get('compressed', False):
        print("=== DECOMPRESSING DATA ===")
        x_train_b64 = context['x_train_compressed']
        y_train_b64 = context['y_train_compressed']
        x_train_compressed = base64.b64decode(x_train_b64.encode('utf-8'))
        y_train_compressed = base64.b64decode(y_train_b64.encode('utf-8'))
        x_train_json = gzip.decompress(x_train_compressed).decode('utf-8')
        y_train_json = gzip.decompress(y_train_compressed).decode('utf-8')
        x_train = np.array(json.loads(x_train_json))
        y_train = np.array(json.loads(y_train_json))
        print(f"Decompressed data - x_train: {{x_train.shape}}, y_train: {{y_train.shape}}")
    else:
        print("=== USING UNCOMPRESSED DATA ===")
        if 'x_train' not in context:
            raise ValueError("Missing x_train in context")
        if 'y_train' not in context:
            raise ValueError("Missing y_train in context")
        x_train = np.array(context['x_train'])
        y_train = np.array(context['y_train'])
        print(f"Uncompressed data - x_train: {{x_train.shape}}, y_train: {{y_train.shape}}")
    
    if x_train.size == 0 or y_train.size == 0:
        raise ValueError("Training data is empty")
    
    print(f"Training data loaded: x_train={{x_train.shape}}, y_train={{y_train.shape}}")
    
    # ENHANCED: Data validation with y_train analysis
    print("=== ENHANCED DATA VALIDATION ===")
    print(f"x_train range: [{{x_train.min():.3f}}, {{x_train.max():.3f}}]")
    print(f"x_train dtype: {{x_train.dtype}}")
    print(f"y_train shape: {{y_train.shape}}, dtype: {{y_train.dtype}}")
    
    # Comprehensive y_train analysis
    if len(y_train.shape) == 1:
        print(f"y_train: integer labels, unique: {{np.unique(y_train)}}")
        print(f"y_train range: [{{y_train.min()}}, {{y_train.max()}}]")
    else:
        print(f"y_train: one-hot encoded, classes: {{y_train.shape[1]}}")
        # Check if it's properly one-hot
        row_sums = np.sum(y_train, axis=1)
        print(f"y_train row sums - min: {{row_sums.min()}}, max: {{row_sums.max()}}, mean: {{row_sums.mean():.3f}}")
        print(f"y_train first 5 rows:\\n{{y_train[:5]}}")
        
        # Check for any anomalies in one-hot encoding
        expected_sum = 1.0
        bad_rows = np.where(np.abs(row_sums - expected_sum) > 0.01)[0]
        if len(bad_rows) > 0:
            print(f"WARNING: {{len(bad_rows)}} rows with unexpected sums: {{bad_rows[:10]}}")
        else:
            print("✓ All rows have proper one-hot encoding")
    
    # Build model
    def build_model():
        print("=== BUILDING MODEL WITH ENHANCED LOGGING ===")
        model_config = context['model_config']
        dataset_config = context['dataset_config']
        
        print(f"Model config keys: {{list(model_config.keys())}}")
        print(f"Dataset config keys: {{list(dataset_config.keys())}}")
        
        if model_config['data_type'] == 'text':
            print("Building text model...")
            model = keras.Sequential([
                keras.layers.Input(shape=(model_config['sequence_length'],)),
                keras.layers.Embedding(model_config['vocab_size'], model_config['embedding_dim']),
                keras.layers.LSTM(model_config['lstm_units']),
                keras.layers.Dense(model_config['first_hidden_layer_nodes'], activation='relu'),
                keras.layers.Dropout(model_config['first_hidden_layer_dropout']),
                keras.layers.Dense(dataset_config['num_classes'], activation='softmax')
            ])
        else:
            print("Building CNN model...")
            layers = []
            layers.append(keras.layers.Input(shape=dataset_config['input_shape']))
            
            for i in range(model_config['num_layers_conv']):
                if model_config['activation'] == 'leaky_relu':
                    layers.append(keras.layers.Conv2D(
                        model_config['filters_per_conv_layer'],
                        model_config['kernel_size'],
                        activation=None,
                        padding='same'
                    ))
                    layers.append(keras.layers.LeakyReLU(alpha=0.01))
                else:
                    layers.append(keras.layers.Conv2D(
                        model_config['filters_per_conv_layer'],
                        model_config['kernel_size'],
                        activation=model_config['activation'],
                        padding='same'
                    ))
                layers.append(keras.layers.MaxPooling2D((2, 2)))
            
            layers.append(keras.layers.Flatten())
            layers.append(keras.layers.Dense(model_config['first_hidden_layer_nodes'], activation='relu'))
            layers.append(keras.layers.Dropout(model_config['first_hidden_layer_dropout']))
            layers.append(keras.layers.Dense(dataset_config['num_classes'], activation='softmax'))
            model = keras.Sequential(layers)
        
        print(f"Model built with {{len(model.layers)}} layers")
        return model

    print("=== COMPILING MODEL WITH ENHANCED DIAGNOSTICS ===")
    model = build_model()
    
    # Test both metric names to see which one works
    print("Testing metric name compatibility...")
    
    # Try compiling with 'accuracy' first
    try:
        test_model_1 = keras.models.clone_model(model)
        test_model_1.compile(
            optimizer=context['model_config']['optimizer'],
            loss=context['model_config']['loss'],
            metrics=['accuracy']
        )
        print("✓ 'accuracy' metric compilation successful")
        accuracy_works = True
    except Exception as e:
        print(f"✗ 'accuracy' metric compilation failed: {{e}}")
        accuracy_works = False
    
    # Try compiling with 'categorical_accuracy'
    try:
        test_model_2 = keras.models.clone_model(model)
        test_model_2.compile(
            optimizer=context['model_config']['optimizer'],
            loss=context['model_config']['loss'],
            metrics=['categorical_accuracy']
        )
        print("✓ 'categorical_accuracy' metric compilation successful")
        categorical_accuracy_works = True
    except Exception as e:
        print(f"✗ 'categorical_accuracy' metric compilation failed: {{e}}")
        categorical_accuracy_works = False
    
    # Use the working metric for actual training
    if categorical_accuracy_works:
        metric_to_use = 'categorical_accuracy'
        print(f"Using 'categorical_accuracy' for training")
    elif accuracy_works:
        metric_to_use = 'accuracy'
        print(f"Using 'accuracy' for training")
    else:
        metric_to_use = 'accuracy'  # Fallback
        print(f"WARNING: Neither metric worked perfectly, using 'accuracy' as fallback")
    
    model.compile(
        optimizer=context['model_config']['optimizer'],
        loss=context['model_config']['loss'],
        metrics=[metric_to_use]
    )
    print(f"Model compiled with '{{metric_to_use}}' metric - Total params: {{model.count_params()}}")
    print(f"Model metric names: {{model.metrics_names}}")
    
    # 🎯 PHASE 4: IMPLEMENT MANUAL VALIDATION SPLIT FOR ACTUAL TRAINING
    print("=== PHASE 4: IMPLEMENTING MANUAL VALIDATION SPLIT FOR ACTUAL TRAINING ===")
    validation_split_val = {validation_split_value}
    
    if validation_split_val > 0:
        print(f"🔧 APPLYING MANUAL VALIDATION SPLIT: {{validation_split_val}}")
        
        # Calculate split index
        split_idx = int(len(x_train) * (1 - validation_split_val))
        
        # Manual split
        x_train_manual = x_train[:split_idx]
        y_train_manual = y_train[:split_idx]
        x_val_manual = x_train[split_idx:]
        y_val_manual = y_train[split_idx:]
        
        print(f"Manual validation split applied:")
        print(f"  - Training samples: {{len(x_train_manual)}} ({{(1-validation_split_val)*100:.1f}}%)")
        print(f"  - Validation samples: {{len(x_val_manual)}} ({{validation_split_val*100:.1f}}%)")
        print(f"  - Train data: x={{x_train_manual.shape}}, y={{y_train_manual.shape}}")
        print(f"  - Val data: x={{x_val_manual.shape}}, y={{y_val_manual.shape}}")
        
        # Verify split integrity
        print("Verifying split integrity...")
        print(f"  - Original total: {{len(x_train)}}")
        print(f"  - Split total: {{len(x_train_manual) + len(x_val_manual)}}")
        print(f"  - Split matches: {{len(x_train) == len(x_train_manual) + len(x_val_manual)}}")
        
        # Train with manual validation_data parameter (BYPASSES THE BUG)
        print("🚀 TRAINING WITH MANUAL VALIDATION_DATA (BUG BYPASS)...")
        training_params = {{
            'x': x_train_manual,
            'y': y_train_manual,
            'epochs': context['model_config']['epochs'],
            'validation_data': (x_val_manual, y_val_manual),  # 🎯 KEY FIX: Use validation_data instead of validation_split
            'verbose': 1
        }}
        
    else:
        print("🔧 NO VALIDATION SPLIT REQUESTED")
        print("Training without validation...")
        training_params = {{
            'x': x_train,
            'y': y_train,
            'epochs': context['model_config']['epochs'],
            'verbose': 1
        }}
    
    print(f"Training parameters: {{list(training_params.keys())}}")
    
    # Enhanced callback to monitor validation metrics
    class EnhancedMetricsLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {{}}
            print(f"Epoch {{epoch+1}} ENHANCED logs:")
            
            # Log all available metrics
            for key, value in logs.items():
                print(f"  {{key}}: {{value}}")
            
            # Specifically check validation accuracy
            val_acc_key = None
            for key in logs.keys():
                if 'val' in key and ('acc' in key or 'accuracy' in key):
                    val_acc_key = key
                    break
            
            if val_acc_key:
                val_acc_value = logs[val_acc_key]
                print(f"  >>> VALIDATION ACCURACY ({{val_acc_key}}): {{val_acc_value}} <<<")
                if val_acc_value == 0.0:
                    print(f"  >>> WARNING: Validation accuracy is ZERO! <<<")
                else:
                    print(f"  >>> GOOD: Validation accuracy is non-zero <<<")
            else:
                print(f"  >>> INFO: No validation accuracy found (training without validation) <<<")
                print(f"  >>> Available keys: {{list(logs.keys())}} <<<")
    
    print("x_train shape: %s", x_train.shape)
    print("y_train shape: %s", y_train.shape)
    print("validation_split: %s", validation_split_val)
    print("Unique y_train classes: %s", np.unique(np.argmax(y_train, axis=1)))
   
    # Execute training with prepared parameters
    print("🚀 STARTING TRAINING WITH MANUAL VALIDATION SPLIT FIX...")
    history = model.fit(
        **training_params,
        callbacks=[EnhancedMetricsLogger()]
    )
    
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    print("=== TRAINING COMPLETED ===")
    print(f"Training time: {{training_time:.2f}} seconds")
    
    # COMPREHENSIVE ANALYSIS of training results
    print("=== COMPREHENSIVE TRAINING RESULTS ANALYSIS ===")
    print(f"history type: {{type(history)}}")
    print(f"history.history type: {{type(history.history)}}")
    print(f"history.history keys: {{list(history.history.keys())}}")
    
    print("model.history keys: %s", history.history.keys())
    # Handle both metric naming conventions
    val_acc_key = None
    for key in history.history.keys():
        if 'val' in key and ('acc' in key or 'accuracy' in key):
            val_acc_key = key
            break

    if val_acc_key:
        print("Final epoch validation accuracy (%s): %s", val_acc_key, history.history[val_acc_key][-1])
    else:
        print("No validation accuracy metric found in history")
    
    # Detailed analysis of each metric
    for key, values in history.history.items():
        print(f"\\n{{key}} analysis:")
        print(f"  Length: {{len(values)}}")
        print(f"  Type: {{type(values)}}")
        print(f"  Values: {{values}}")
        
        if len(values) > 0:
            print(f"  First: {{values[0]}}")
            print(f"  Last: {{values[-1]}}")
            print(f"  Min: {{min(values)}}")
            print(f"  Max: {{max(values)}}")
            print(f"  All zeros: {{all(v == 0 for v in values)}}")
            print(f"  Any non-zero: {{any(v != 0 for v in values)}}")
    
    # 🎯 PHASE 4: VALIDATION SPLIT FIX VERIFICATION
    print("=== PHASE 4: VALIDATION SPLIT FIX VERIFICATION ===")
    
    # Check if manual validation split fixed the issue
    val_metrics_found = [k for k in history.history.keys() if 'val' in k]
    val_acc_metrics = [k for k in val_metrics_found if 'acc' in k or 'accuracy' in k]
    
    manual_split_success = False
    if val_acc_metrics:
        for val_acc_key in val_acc_metrics:
            val_acc_values = history.history[val_acc_key]
            if val_acc_values and any(v > 0 for v in val_acc_values):
                manual_split_success = True
                print(f"✅ MANUAL VALIDATION SPLIT FIX SUCCESSFUL!")
                print(f"✅ Key '{{val_acc_key}}' has non-zero values: {{val_acc_values}}")
                break
        
        if not manual_split_success:
            print(f"❌ MANUAL VALIDATION SPLIT STILL FAILING")
            print(f"❌ All validation accuracy metrics still zero")
            for val_acc_key in val_acc_metrics:
                print(f"❌ {{val_acc_key}}: {{history.history[val_acc_key]}}")
    else:
        if validation_split_val > 0:
            print(f"❌ NO VALIDATION METRICS FOUND (expected with validation_split={{validation_split_val}})")
        else:
            print(f"✓ NO VALIDATION METRICS FOUND (as expected with no validation split)")
    
    # Build comprehensive result with all environment info
    result = {{
        'history': history.history,
        'model_params': int(model.count_params()),
        'training_time': float(training_time),
        'epochs_completed': int(len(history.history.get('loss', []))),
        'model_summary': {{
            'total_layers': int(len(model.layers)),
            'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
            'non_trainable_params': int(model.count_params() - sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
        }},
        # COMPREHENSIVE ENVIRONMENT INFO
        'environment_info': {{
            'python_version': sys.version,
            'tensorflow_version': tf.__version__,
            'tensorflow_keras_version': tf.keras.__version__,
            'tensorflow_cuda_built': tf.test.is_built_with_cuda(),
            'cuda_version': tf.sysconfig.get_build_info().get('cuda_version', 'Unknown'),
            'cudnn_version': tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown'),
            'gpu_devices': str(tf.config.list_physical_devices('GPU')),
            'keras_standalone_available': keras_standalone,
            'keras_standalone_version': keras_version,
            'tf_build_info': build_info,
            'metric_used': metric_to_use,
            'metric_names': model.metrics_names,
            'validation_split_used': validation_split_val,
            'manual_validation_split_applied': validation_split_val > 0  # 🎯 PHASE 4 INDICATOR
        }},
        # 🎯 PHASE 4: VALIDATION SPLIT FIX ANALYSIS
        'validation_split_fix_analysis': {{
            'validation_split_value': validation_split_val,
            'manual_split_applied': validation_split_val > 0,
            'validation_metrics_found': val_metrics_found,
            'validation_accuracy_keys': val_acc_metrics,
            'manual_split_success': manual_split_success,
            'training_method': 'validation_data' if validation_split_val > 0 else 'no_validation',
            'bug_bypass_confirmed': manual_split_success and validation_split_val > 0
        }}
    }}
    
    # Convert arrays to Python floats for JSON serialization
    for key, values in result['history'].items():
        if values and isinstance(values, list):
            result['history'][key] = [float(v) if v is not None else None for v in values]
    
    print("=== COMPREHENSIVE RESULT PREPARED WITH PHASE 4 FIX ===")
    print(f"Result keys: {{list(result.keys())}}")
    print(f"Environment info keys: {{list(result['environment_info'].keys())}}")
    print(f"Phase 4 fix analysis: {{result['validation_split_fix_analysis']}}")
    
except Exception as e:
    print("=== ERROR OCCURRED ===")
    print(f"Error: {{e}}")
    traceback.print_exc()
    
    # FIXED: Build error result with proper variable handling
    error_env_info = {{
        'error_during_diagnostics': str(e),
        'tf_available': keras_available,
    }}
    
    # Only add these if they were successfully determined
    if keras_available:
        try:
            error_env_info.update({{
                'tensorflow_version': tf.__version__,
                'tensorflow_keras_version': tf.keras.__version__,
            }})
        except:
            pass
    
    if keras_standalone:
        error_env_info['keras_standalone_version'] = keras_version
    
    result = {{
        'history': {{}},
        'model_params': 0,
        'training_time': 0.0,
        'epochs_completed': 0,
        'error': str(e),
        'success': False,
        'environment_info': error_env_info,
        'validation_split_fix_analysis': {{
            'manual_split_applied': False,
            'error_occurred': True
        }}
    }}

print("=== EXECUTION COMPLETE ===")
"""
    
    return training_code


# Additional utility functions for GPU proxy code generation
def generate_test_validation_split_code(validation_split_value: float) -> str:
    """Generate code specifically for testing validation split behavior."""
    return f"""
# Validation Split Test Code
validation_split_val = {validation_split_value}
print(f"Testing validation split with value: {{validation_split_val}}")

# Test implementation would go here
"""


def generate_model_building_code() -> str:
    """Generate the model building portion of the GPU proxy code."""
    return """
def build_model():
    print("=== BUILDING MODEL ===")
    model_config = context['model_config']
    dataset_config = context['dataset_config']
    
    if model_config['data_type'] == 'text':
        # Text model implementation
        pass
    else:
        # CNN model implementation
        pass
    
    return model
"""


def generate_training_execution_code() -> str:
    """Generate the training execution portion of the GPU proxy code."""
    return """
# Execute training with manual validation split
if validation_split_val > 0:
    # Manual validation split implementation
    pass
else:
    # No validation training
    pass
"""


if __name__ == "__main__":
    # Test the code generation
    validation_split = 0.2
    code = get_gpu_proxy_training_code(validation_split)
    print("Generated GPU proxy training code:")
    print(f"Code length: {len(code)} characters")
    print("First 500 characters:")
    print(code[:500])