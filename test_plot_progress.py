#!/usr/bin/env python3
"""
Test script for plot generation progress tracking

This script tests the new plot progress callback functionality by running
a short optimization with minimal trials and monitoring the progress updates.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from optimizer import ModelOptimizer, OptimizationConfig, OptimizationMode, OptimizationObjective
import logging

# Set up logging to see progress updates
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_plot_progress():
    """Test the plot progress tracking functionality"""
    print("🧪 Testing Plot Progress Tracking")
    print("=" * 50)
    
    # Create a minimal optimization config for testing
    config = OptimizationConfig(
        dataset_name="mnist",  # Required field
        optimize_for="accuracy",  # Required field
        mode=OptimizationMode.SIMPLE,
        objective=OptimizationObjective.VAL_ACCURACY,
        trials=2,  # Minimal trials for quick testing
        max_epochs_per_trial=2,  # Minimal epochs for quick testing
        min_epochs_per_trial=1,
        health_weight=0.5,  # Required field
        validation_split=0.2,
        use_runpod_service=False,  # Local execution only for testing
        timeout_hours=None
    )
    
    print(f"📋 Config: {config.trials} trials, {config.max_epochs_per_trial} epochs each")
    
    # Test progress callback to capture updates
    progress_updates = []
    
    def test_progress_callback(progress_data):
        """Capture progress updates for analysis"""
        progress_updates.append(progress_data)
        
        # Log plot generation progress if available
        if hasattr(progress_data, 'plot_generation') and progress_data.plot_generation:
            plot_gen = progress_data.plot_generation
            print(f"🎨 PLOT PROGRESS - Trial {progress_data.trial_number}: {plot_gen.get('current_plot', 'Unknown')} "
                  f"({plot_gen.get('completed_plots', 0)}/{plot_gen.get('total_plots', 0)}) "
                  f"[{plot_gen.get('plot_progress', 0)*100:.1f}%] - Status: {plot_gen.get('status', 'unknown')}")
        elif hasattr(progress_data, 'current_epoch') and progress_data.current_epoch:
            print(f"🔄 EPOCH PROGRESS - Trial {progress_data.trial_number}: Epoch {progress_data.current_epoch}/{progress_data.total_epochs or 'N/A'}")
    
    try:
        # Create optimizer with test dataset (MNIST for speed)
        optimizer = ModelOptimizer("mnist", config, progress_callback=test_progress_callback)
        print(f"✅ Created optimizer for dataset: mnist")
        
        # Run optimization
        print("\n🚀 Starting optimization...")
        result = optimizer.optimize()
        
        print(f"\n📊 Optimization completed!")
        print(f"Best accuracy: {result.best_performance.get('test_accuracy', 'N/A'):.4f}")
        
        # Analyze captured progress updates
        print(f"\n📈 Analysis of {len(progress_updates)} progress updates:")
        
        plot_progress_updates = []
        epoch_progress_updates = []
        
        for update in progress_updates:
            if hasattr(update, 'plot_generation') and update.plot_generation:
                plot_progress_updates.append(update)
            elif hasattr(update, 'current_epoch') and update.current_epoch:
                epoch_progress_updates.append(update)
        
        print(f"  - Epoch progress updates: {len(epoch_progress_updates)}")
        print(f"  - Plot progress updates: {len(plot_progress_updates)}")
        
        if plot_progress_updates:
            print("\n🎨 Plot Progress Details:")
            for i, update in enumerate(plot_progress_updates[:10]):  # Show first 10
                plot_gen = update.plot_generation
                print(f"  {i+1}. Trial {update.trial_number}: {plot_gen.get('current_plot', 'Unknown')} "
                      f"({plot_gen.get('completed_plots', 0)}/{plot_gen.get('total_plots', 0)}) "
                      f"- {plot_gen.get('status', 'unknown')}")
        
        if len(plot_progress_updates) > 0:
            print("✅ SUCCESS: Plot progress tracking is working!")
        else:
            print("⚠️  WARNING: No plot progress updates detected")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plot_progress()