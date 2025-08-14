"""
Configuration Comparison Diagnostic for GPU vs Local Accuracy Gap

This script will help identify differences in configuration parameters
between RunPod GPU execution and local CPU execution that could explain
the 6% accuracy discrepancy.
"""

import sys
from pathlib import Path
import json
from dataclasses import asdict, fields

# Add src to path
current_file = Path(__file__)
project_root = current_file.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from optimizer import OptimizationConfig, OptimizationMode, OptimizationObjective
from model_builder import ModelConfig
from dataset_manager import DatasetManager
from utils.logger import logger

def analyze_config_differences():
    """
    Analyze configuration differences between GPU and Local execution paths
    """
    logger.debug("running analyze_config_differences ... Starting configuration analysis")
    
    # Create sample configurations that match your testing
    opt_config = OptimizationConfig(
        mode=OptimizationMode.SIMPLE,
        objective=OptimizationObjective.VAL_ACCURACY,
        n_trials=3,
        use_runpod_service=True,
        gpu_proxy_sample_percentage=1.0,
        validation_split=0.2,
        max_training_time_minutes=60.0,
        max_epochs_per_trial=20,
        min_epochs_per_trial=5
    )
    
    # Create ModelConfig (what local execution gets)
    model_config = ModelConfig()
    
    print("=" * 80)
    print("CONFIGURATION COMPARISON ANALYSIS")
    print("=" * 80)
    
    # 1. Analyze current parameter transfer
    print("\n1. CURRENT PARAMETER TRANSFER ANALYSIS")
    print("-" * 50)
    
    config_to_model_params = [
        'gpu_proxy_sample_percentage',
        'validation_split',
    ]
    
    print(f"Currently transferred parameters: {len(config_to_model_params)}")
    for param_name in config_to_model_params:
        if hasattr(opt_config, param_name) and hasattr(model_config, param_name):
            opt_value = getattr(opt_config, param_name)
            model_value = getattr(model_config, param_name)
            print(f"  ✅ {param_name}: {opt_value} -> {model_value}")
        else:
            print(f"  ❌ {param_name}: MISSING in one of the configs")
    
    # 2. Analyze OptimizationConfig parameters NOT being transferred
    print("\n2. OPTIMIZATION CONFIG PARAMETERS NOT TRANSFERRED")
    print("-" * 50)
    
    opt_config_fields = {f.name: f.type for f in fields(OptimizationConfig)}
    model_config_fields = {f.name: f.type for f in fields(ModelConfig)}
    
    missing_transfers = []
    for param_name, param_type in opt_config_fields.items():
        if param_name in model_config_fields and param_name not in config_to_model_params:
            opt_value = getattr(opt_config, param_name)
            model_value = getattr(model_config, param_name)
            if opt_value != model_value:
                missing_transfers.append((param_name, opt_value, model_value))
                print(f"  ⚠️  {param_name}: OptConfig={opt_value}, ModelConfig={model_value}")
    
    if not missing_transfers:
        print("  ✅ No missing parameter transfers found")
    else:
        print(f"  🚨 Found {len(missing_transfers)} parameters with different default values!")
    
    # 3. Analyze JSON payload vs full local configuration
    print("\n3. JSON PAYLOAD vs LOCAL CONFIGURATION")
    print("-" * 50)
    
    # Simulate the JSON payload sent to RunPod
    json_payload = {
        "validation_split": opt_config.validation_split,
        "max_training_time": opt_config.max_training_time_minutes,
        "mode": opt_config.mode.value,
        "objective": opt_config.objective.value,
        "gpu_proxy_sample_percentage": opt_config.gpu_proxy_sample_percentage
    }
    
    print("JSON payload sent to RunPod:")
    for key, value in json_payload.items():
        print(f"  📤 {key}: {value}")
    
    print(f"\nJSON payload size: {len(json_payload)} parameters")
    
    # Full model config for local
    model_config_dict = asdict(model_config)
    print(f"Local ModelConfig size: {len(model_config_dict)} parameters")
    
    print(f"\nParameter coverage: {len(json_payload)/len(model_config_dict)*100:.1f}%")
    
    # 4. Critical parameter analysis
    print("\n4. CRITICAL PARAMETERS THAT COULD AFFECT TRAINING")
    print("-" * 50)
    
    critical_params = [
        'epochs', 'optimizer', 'learning_rate', 'loss', 'metrics',
        'batch_normalization', 'activation', 'kernel_initializer',
        'num_layers_conv', 'filters_per_conv_layer', 'kernel_size',
        'num_layers_hidden', 'first_hidden_layer_nodes',
        'enable_gradient_clipping', 'gradient_clip_norm'
    ]
    
    for param in critical_params:
        if hasattr(model_config, param):
            value = getattr(model_config, param)
            print(f"  🎯 {param}: {value}")
        else:
            print(f"  ❌ {param}: NOT FOUND in ModelConfig")
    
    # 5. Training execution path differences
    print("\n5. TRAINING EXECUTION PATH ANALYSIS")
    print("-" * 50)
    
    print("Local execution path:")
    print("  optimizer.py -> _train_locally_for_trial() -> ModelBuilder.train() -> _train_locally_optimized()")
    print("  Uses: Manual validation split with validation_data parameter")
    print("  Uses: Full ModelConfig with all parameters")
    
    print("\nRunPod execution path:")
    print("  optimizer.py -> _train_via_runpod_service() -> JSON API -> handler.py -> create_and_train_model()")
    print("  Uses: Limited JSON configuration")
    print("  Uses: ModelConfig defaults + JSON overrides")
    
    # 6. Validation split implementation
    print("\n6. VALIDATION SPLIT IMPLEMENTATION")
    print("-" * 50)
    
    print("Both paths claim to use manual validation split, but:")
    print("  Local: ModelBuilder._train_locally_optimized() with validation_data")
    print("  RunPod: Unknown - depends on handler.py implementation")
    print("  ⚠️  This could be a key difference!")
    
    return missing_transfers, json_payload, model_config_dict

def suggest_fixes(missing_transfers, json_payload, model_config_dict):
    """
    Suggest specific fixes for the configuration differences
    """
    print("\n" + "=" * 80)
    print("RECOMMENDED FIXES")
    print("=" * 80)
    
    if missing_transfers:
        print("\n1. EXPAND PARAMETER TRANSFER")
        print("-" * 30)
        print("Add these parameters to config_to_model_params in optimizer.py:")
        print("```python")
        print("config_to_model_params = [")
        print("    'gpu_proxy_sample_percentage',")
        print("    'validation_split',")
        for param_name, opt_value, model_value in missing_transfers:
            print(f"    '{param_name}',  # {opt_value} vs {model_value}")
        print("]")
        print("```")
    
    print("\n2. EXPAND JSON PAYLOAD")
    print("-" * 25)
    print("Consider adding these critical parameters to the RunPod JSON payload:")
    
    critical_missing = []
    critical_params = ['epochs', 'optimizer', 'learning_rate', 'activation', 'batch_normalization']
    
    for param in critical_params:
        if param not in json_payload:
            critical_missing.append(param)
    
    if critical_missing:
        print("```python")
        print('request_payload = {')
        print('    "input": {')
        print('        "command": "start_training",')
        print('        # ... existing fields ...')
        print('        "config": {')
        for key, value in json_payload.items():
            print(f'            "{key}": {repr(value)},')
        for param in critical_missing:
            print(f'            "{param}": self.model_config.{param},  # ADD THIS')
        print('        }')
        print('    }')
        print('}')
        print("```")
    
    print("\n3. VERIFICATION STEPS")
    print("-" * 20)
    print("To verify the fix:")
    print("1. Add comprehensive logging to both execution paths")
    print("2. Compare ModelConfig state before training in both environments")
    print("3. Verify identical training parameters are used")
    print("4. Check validation split implementation consistency")

if __name__ == "__main__":
    try:
        missing_transfers, json_payload, model_config_dict = analyze_config_differences()
        suggest_fixes(missing_transfers, json_payload, model_config_dict)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print("Review the findings above to identify the source of the 6% accuracy gap.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()