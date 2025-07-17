"""
Model Optimizer Router

Routes optimization requests to the appropriate optimizer implementation:
- 'simple': Pure accuracy-focused optimization (original approach)
- 'health': Health-aware optimization (new approach)

Provides unified interface for both optimization strategies.
"""

from datetime import datetime
import sys
from typing import Dict, Any
from utils.logger import logger
from model_builder import create_and_train_model
from model_optimizer_simple import optimize_model as simple_optimize
from model_optimizer_health import optimize_model as health_optimize


def optimize_model(
    dataset_name: str,
    optimizer: str = 'health',
    optimize_for: str = "accuracy", 
    trials: int = 50,
    create_model: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Route optimization to the appropriate optimizer implementation
    
    Args:
        dataset_name: Name of dataset to optimize
        optimizer: Optimizer type ('simple' or 'health')
        optimize_for: Optimization objective
        trials: Number of trials to run
        create_model: Whether to build a model with optimized parameters
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing:
        - 'optimization_result': OptimizationResult from the selected optimizer
        - 'model_result': Dict from create_and_train_model (if create_model=True)
        - 'run_name': The unified run name used
        - 'best_value': Best optimization value (convenience accessor)
        - 'best_params': Best parameters (convenience accessor)
    """
    logger.debug(f"running optimize_model (router) ... Routing request to '{optimizer}' optimizer")
    logger.debug(f"running optimize_model (router) ... Dataset: {dataset_name}")
    logger.debug(f"running optimize_model (router) ... Objective: {optimize_for}")
    logger.debug(f"running optimize_model (router) ... Trials: {trials}")
    logger.debug(f"running optimize_model (router) ... Create model: {create_model}")
    
    # CREATE UNIFIED RUN NAME HERE - one place, used everywhere
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    dataset_clean = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    
    if optimizer == 'health':
        run_name = f"{timestamp}_{dataset_clean}_health"
    elif optimizer == 'simple':
        run_name = f"{timestamp}_{dataset_clean}_simple-{optimize_for}"
    else:
        run_name = f"{timestamp}_{dataset_clean}_{optimizer}"
    
    logger.debug(f"running optimize_model (router) ... Using unified run name: {run_name}") 
    
    # Run optimization
    optimization_result = None
    if optimizer == 'simple':
        logger.debug(f"running optimize_model (router) ... Loading simple optimizer...")
        try:            
            logger.debug(f"running optimize_model (router) ... Simple optimizer loaded successfully")
            optimization_result = simple_optimize(
                dataset_name=dataset_name,
                optimize_for=optimize_for,
                trials=trials,
                run_name=run_name,
                **kwargs
            )
        except ImportError as e:
            logger.error(f"running optimize_model (router) ... Failed to import simple optimizer: {e}")
            raise ImportError(f"Simple optimizer not available: {e}")
            
    elif optimizer == 'health':
        logger.debug(f"running optimize_model (router) ... Loading health optimizer...")
        try:
            
            logger.debug(f"running optimize_model (router) ... Health optimizer loaded successfully")
            optimization_result = health_optimize(
                dataset_name=dataset_name,
                optimize_for=optimize_for,
                trials=trials,
                run_name=run_name,
                **kwargs
            )
        except ImportError as e:
            logger.error(f"running optimize_model (router) ... Failed to import health optimizer: {e}")
            raise ImportError(f"Health optimizer not available: {e}")
        except NotImplementedError as e:
            logger.error(f"running optimize_model (router) ... Health optimizer not implemented yet: {e}")
            raise NotImplementedError(f"Health optimizer coming soon: {e}")
            
    else:
        available_optimizers = ['simple', 'health']
        logger.error(f"running optimize_model (router) ... Unknown optimizer: {optimizer}")
        raise ValueError(f"Unknown optimizer '{optimizer}'. Available: {available_optimizers}")
    
    # Prepare return value
    result = {
        'optimization_result': optimization_result,
        'run_name': run_name,
        'best_value': optimization_result.best_value,
        'best_params': optimization_result.best_params,
        'model_result': None
    }
    
    # Optionally build model with optimized parameters
    if create_model:
        logger.debug(f"running optimize_model (router) ... Building model with optimized parameters...")
        logger.debug(f"running optimize_model (router) ... Using run name: {run_name}")
        
        try:
            model_result = create_and_train_model(
                dataset_name=dataset_name,
                run_name=run_name,
                **optimization_result.best_params
            )            
            result['model_result'] = model_result
            logger.debug(f"running optimize_model (router) ... ✅ Model built successfully!")
            logger.debug(f"running optimize_model (router) ... Model saved to: {model_result.get('model_path', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"running optimize_model (router) ... Model building failed: {e}")
            # Don't fail the whole operation, just log the error
            result['model_result'] = {'error': str(e)}
    
    return result



if __name__ == "__main__":
    # Parse command line arguments
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    
    # Extract optimizer type (new parameter)
    optimizer = args.get('optimizer', 'simple')  # Default to simple for backwards compatibility
    dataset_name = args.get('dataset', 'mnist')  # Default dataset
    optimize_for = args.get('optimize_for', 'accuracy')
    trials = int(args.get('trials', '50'))
    create_model = args.get('create_model', 'true').lower() in ['true', '1', 'yes', 'on']  # Default to True
    
    # Convert parameters (same logic as in the individual optimizers)
    # Integer parameters
    int_params = [
        'n_trials', 'n_startup_trials', 'n_warmup_steps', 'random_seed',
        'max_epochs_per_trial', 'early_stopping_patience'
    ]
    for int_param in int_params:
        if int_param in args:
            try:
                args[int_param] = int(args[int_param])
                logger.debug(f"running model_optimizer.py (router) ... Converted {int_param} to int: {args[int_param]}")
            except ValueError:
                logger.warning(f"running model_optimizer.py (router) ... Invalid {int_param}: {args[int_param]}, using default")
                del args[int_param]
    
    # Float parameters
    float_params = [
        'timeout_hours', 'max_training_time_minutes', 'validation_split', 'test_size'
    ]
    for float_param in float_params:
        if float_param in args:
            try:
                args[float_param] = float(args[float_param])
                logger.debug(f"running model_optimizer.py (router) ... Converted {float_param} to float: {args[float_param]}")
            except ValueError:
                logger.warning(f"running model_optimizer.py (router) ... Invalid {float_param}: {args[float_param]}, using default")
                del args[float_param]
    
    # Boolean parameters
    bool_params = [
        'save_best_model', 'save_optimization_history', 'create_comparison_plots',
        'enable_early_stopping'
    ]
    for bool_param in bool_params:
        if bool_param in args:
            args[bool_param] = args[bool_param].lower() in ['true', '1', 'yes', 'on']
            logger.debug(f"running model_optimizer.py (router) ... Converted {bool_param} to bool: {args[bool_param]}")
    
    # Remove router-specific parameters before passing to optimizer
    router_params = {'optimizer', 'dataset', 'optimize_for', 'trials', 'create_model'}
    optimizer_args = {k: v for k, v in args.items() if k not in router_params}
    
    logger.debug(f"running model_optimizer.py (router) ... Starting optimization")
    logger.debug(f"running model_optimizer.py (router) ... Optimizer: {optimizer}")
    logger.debug(f"running model_optimizer.py (router) ... Dataset: {dataset_name}")
    logger.debug(f"running model_optimizer.py (router) ... Objective: {optimize_for}")
    logger.debug(f"running model_optimizer.py (router) ... Trials: {trials}")
    logger.debug(f"running model_optimizer.py (router) ... Create model: {create_model}")
    logger.debug(f"running model_optimizer.py (router) ... Additional args: {optimizer_args}")
    
    try:
        # Single function call handles both optimization and model building
        result = optimize_model(
            dataset_name=dataset_name,
            optimizer=optimizer,
            optimize_for=optimize_for,
            trials=trials,
            create_model=create_model,
            **optimizer_args
        )
        
        # Print optimization results
        print(result['optimization_result'].summary())
        
        logger.debug(f"running model_optimizer.py (router) ... ✅ Optimization completed successfully via {optimizer} optimizer!")
        
        # Print final summary
        print(f"\n🎉 Complete Optimization Summary:")
        print(f"Dataset: {dataset_name}")
        print(f"Optimizer: {optimizer}")
        print(f"Objective: {optimize_for}")
        print(f"Best {optimize_for}: {result['best_value']:.4f}")
        print(f"Run name: {result['run_name']}")
        
        if create_model and result['model_result'] and 'error' not in result['model_result']:
            print(f"✅ Model built successfully!")
            if result['model_result'].get('model_path'):
                print(f"Model saved to: {result['model_result']['model_path']}")
        elif create_model and result['model_result'] and 'error' in result['model_result']:
            print(f"❌ Model building failed: {result['model_result']['error']}")
        elif not create_model:
            print(f"ℹ️  Model building skipped (create_model=False)")
        
    except Exception as e:
        logger.error(f"running model_optimizer.py (router) ... ❌ Optimization failed: {e}")
        sys.exit(1)