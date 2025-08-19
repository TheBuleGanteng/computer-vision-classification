"""
Multi-GPU Diagnostic Script
Place this in your project root and run: python multi_gpu_diagnostic.py
"""

import tensorflow as tf
from pathlib import Path
import sys

# Add src to path
current_file = Path(__file__)
project_root = current_file.parent
sys.path.insert(0, str(project_root / "src"))

from src.utils.logger import logger

def diagnose_multi_gpu_setup():
    """Diagnose current multi-GPU configuration and utilization"""
    
    logger.debug("running diagnose_multi_gpu_setup ... Starting multi-GPU diagnostic")
    
    # 1. Check available GPUs
    physical_gpus = tf.config.list_physical_devices('GPU')
    logger.debug(f"running diagnose_multi_gpu_setup ... Physical GPUs detected: {len(physical_gpus)}")
    
    for i, gpu in enumerate(physical_gpus):
        logger.debug(f"running diagnose_multi_gpu_setup ... GPU {i}: {gpu}")
        
        # Check memory configuration
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            logger.debug(f"running diagnose_multi_gpu_setup ... GPU {i} details: {gpu_details}")
        except Exception as e:
            logger.debug(f"running diagnose_multi_gpu_setup ... Could not get GPU {i} details: {e}")
    
    # 2. Test MirroredStrategy setup
    if len(physical_gpus) > 1:
        logger.debug("running diagnose_multi_gpu_setup ... Testing MirroredStrategy setup")
        
        strategy = tf.distribute.MirroredStrategy()
        logger.debug(f"running diagnose_multi_gpu_setup ... Strategy devices: {strategy.extended.worker_devices}")
        logger.debug(f"running diagnose_multi_gpu_setup ... Number of replicas: {strategy.num_replicas_in_sync}")
        
        # 3. Test simple computation across GPUs
        with strategy.scope():
            # Create a simple model to test distribution
            test_model = tf.keras.Sequential([ # type: ignore
                tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)), # type: ignore
                tf.keras.layers.Dense(64, activation='relu'), # type: ignore # type: ignore
                tf.keras.layers.Dense(10, activation='softmax') # pyright: ignore[reportAttributeAccessIssue]
            ])
            
            test_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.debug(f"running diagnose_multi_gpu_setup ... Test model created with {test_model.count_params()} parameters")
        
        # 4. Test data distribution
        import numpy as np
        test_x = np.random.random((1000, 100)).astype(np.float32)
        test_y = np.random.randint(0, 10, 1000)
        
        logger.debug("running diagnose_multi_gpu_setup ... Testing data distribution across GPUs")
        
        # Create distributed dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_dataset = test_dataset.batch(32)
        
        # Distribute dataset
        distributed_dataset = strategy.experimental_distribute_dataset(test_dataset)
        logger.debug("running diagnose_multi_gpu_setup ... Dataset distributed successfully")
        
        # 5. Test training step
        @tf.function
        def train_step(inputs):
            features, labels = inputs
            with tf.GradientTape() as tape:
            predictions = test_model(features, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions) # pyright: ignore[reportAttributeAccessIssue]
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, test_model.trainable_variables)

        # TYPE SAFETY: Check if gradients were computed successfully
        if gradients is None:
            raise RuntimeError("Failed to compute gradients - gradient tape returned None")

        # Additional safety check for individual gradient elements
        filtered_gradients = []
        filtered_variables = []
        for grad, var in zip(gradients, test_model.trainable_variables):
            if grad is not None:
                filtered_gradients.append(grad)
                filtered_variables.append(var)

        if not filtered_gradients:
            raise RuntimeError("No valid gradients computed - all gradients are None")

        test_model.optimizer.apply_gradients(zip(filtered_gradients, filtered_variables))
        return loss
        
        @tf.function
        def distributed_train_step(dist_inputs):
            per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        
        # Run a few training steps
        logger.debug("running diagnose_multi_gpu_setup ... Running test training steps")
        
        for i, batch in enumerate(distributed_dataset.take(3)):
            loss = distributed_train_step(batch)
            logger.debug(f"running diagnose_multi_gpu_setup ... Step {i}: loss = {loss:.4f}")
        
        logger.debug("running diagnose_multi_gpu_setup ... Multi-GPU test completed successfully")
        
    else:
        logger.debug("running diagnose_multi_gpu_setup ... Only 1 GPU available, multi-GPU testing skipped")
    
    # 6. Check memory growth settings
    logger.debug("running diagnose_multi_gpu_setup ... Checking GPU memory configuration")
    
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.debug(f"running diagnose_multi_gpu_setup ... Memory growth enabled for {gpu}")
        except Exception as e:
            logger.debug(f"running diagnose_multi_gpu_setup ... Could not set memory growth for {gpu}: {e}")

if __name__ == "__main__":
    diagnose_multi_gpu_setup()