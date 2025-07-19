#!/usr/bin/env python3
"""
Run experiments for the three remaining datasets: ECL, gefcom2014, southern_china
This script focuses only on the datasets that were missing from the previous run.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_model_hyperparams(model, dataset):
    """Get model-specific hyperparameters based on dataset and model"""
    # Hyperparameter configurations for different models and datasets
    hyperparams = {
        # Transformer-based models
        'PatchTST': {'d_model': 16, 'n_heads': 4, 'e_layers': 3, 'd_ff': 128},
        'iTransformer': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
        'Autoformer': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
        'Informer': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
        'Transformer': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
        'Crossformer': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
        # Linear models
        'DLinear': {'d_model': 16, 'n_heads': 4, 'e_layers': 1, 'd_ff': 128},
        'Linear': {'d_model': 16, 'n_heads': 4, 'e_layers': 1, 'd_ff': 128},
        'NLinear': {'d_model': 16, 'n_heads': 4, 'e_layers': 1, 'd_ff': 128},
        'RLinear': {'d_model': 16, 'n_heads': 4, 'e_layers': 1, 'd_ff': 128},
        # CNN-based models
        'TCN': {'d_model': 32, 'n_heads': 4, 'e_layers': 2, 'd_ff': 128},
        'TCN_RevIN': {'d_model': 32, 'n_heads': 4, 'e_layers': 2, 'd_ff': 128},
        # Graph-based models
        'MTGNN': {'d_model': 64, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
        # Advanced models
        'GPT4TS': {'d_model': 256, 'n_heads': 8, 'e_layers': 3, 'd_ff': 512},
        'FSNet': {'d_model': 64, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
        'OneNet': {'d_model': 64, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
        'LIFT': {'d_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
        'LightMTS': {'d_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
    }

    if model in hyperparams:
        return hyperparams[model].values()
    else:
        # Default values for unknown models
        return 16, 4, 2, 128

def run_experiment(dataset, model, pred_len, method, epochs=5, itr=1):
    """Run a single experiment and return results"""
    
    print(f"=== Experiment ===")
    print(f"Running: {dataset} - {model} - {pred_len}h - {method}")
    start_time = time.time()

    # Set model-specific parameters based on dataset and model using hyperparameter configurations
    d_model, n_heads, e_layers, d_ff = get_model_hyperparams(model, dataset)

    # Base command
    cmd = [
        "python", "-u", "run.py",
        "--dataset", dataset,
        "--model", model,
        "--seq_len", "96",
        "--pred_len", str(pred_len),
        "--batch_size", "16",
        "--learning_rate", "0.001",
        "--train_epochs", str(epochs),
        "--itr", str(itr),
        "--features", "M",
        "--d_model", str(d_model),
        "--n_heads", str(n_heads),
        "--e_layers", str(e_layers),
        "--d_ff", str(d_ff)
    ]

    # Add method-specific parameters
    if method == "Offline":
        # Standard offline training
        pass
    elif method == "Online":
        cmd.extend([
            "--online_method", "Online",
            "--online_learning_rate", "0.0001",
            "--only_test",
            "--pretrain"
        ])
    elif method == "FSNet":
        cmd.extend([
            "--online_method", "FSNet",
            "--online_learning_rate", "0.00003",
            "--only_test"
        ])
    elif method == "OneNet":
        cmd.extend([
            "--online_method", "OneNet",
            "--online_learning_rate", "0.0001",
            "--only_test"
        ])
    elif method == "Proceed":
        cmd.extend([
            "--online_method", "Proceed",
            "--online_learning_rate", "0.0001",
            "--only_test"
        ])
    elif method == "ClearE":
        cmd.extend([
            "--online_method", "ClearE",
            "--online_learning_rate", "0.0001",
            "--only_test",
            "--pretrain"
        ])

    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Parse results from output
            output_lines = result.stdout.split('\n')
            mse, mae = None, None
            
            for line in output_lines:
                if 'mse:' in line.lower() and 'mae:' in line.lower():
                    try:
                        parts = line.split(',')
                        mse_part = [p for p in parts if 'mse:' in p.lower()][0]
                        mae_part = [p for p in parts if 'mae:' in p.lower()][0]
                        mse = float(mse_part.split(':')[1].strip())
                        mae = float(mae_part.split(':')[1].strip())
                        break
                    except:
                        continue
            
            if mse is not None and mae is not None:
                print(f"✓ Success: MSE={mse:.4f}, MAE={mae:.4f}")
                return {
                    'dataset': dataset,
                    'model': model,
                    'pred_len': pred_len,
                    'method': method,
                    'mse': mse,
                    'mae': mae,
                    'duration': duration,
                    'success': True,
                    'output': result.stdout
                }
            else:
                print(f"✗ Failed: Could not parse results")
                return {
                    'dataset': dataset,
                    'model': model,
                    'pred_len': pred_len,
                    'method': method,
                    'success': False,
                    'error': "Could not parse results",
                    'output': result.stdout,
                    'duration': duration
                }
        else:
            print(f"✗ Failed: {result.stderr}")
            return {
                'dataset': dataset,
                'model': model,
                'pred_len': pred_len,
                'method': method,
                'success': False,
                'error': result.stderr,
                'output': result.stdout,
                'duration': duration
            }
            
    except subprocess.TimeoutExpired:
        print(f"✗ Failed: Timeout after 1 hour")
        return {
            'dataset': dataset,
            'model': model,
            'pred_len': pred_len,
            'method': method,
            'success': False,
            'error': "Timeout after 1 hour",
            'duration': 3600
        }
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return {
            'dataset': dataset,
            'model': model,
            'pred_len': pred_len,
            'method': method,
            'success': False,
            'error': str(e),
            'duration': time.time() - start_time
        }

def main():
    """Main execution function"""
    print("=== CLEAR-E Remaining Datasets Experiments ===")
    print("Running experiments for: ECL, gefcom2014, southern_china")
    
    # Experimental configuration - only the three remaining datasets
    datasets = ["ECL", "gefcom2014", "southern_china"]
    # All available baseline models for comprehensive comparison
    models = [
        # Transformer-based models
        "PatchTST", "iTransformer", "Autoformer", "Informer", "Transformer", "Crossformer",
        # Linear models
        "DLinear", "Linear", "NLinear", "RLinear",
        # CNN-based models
        "TCN", "TCN_RevIN",
        # Graph-based models
        "MTGNN",
        # Advanced models
        "GPT4TS", "FSNet", "OneNet", "LIFT", "LightMTS"
    ]
    pred_lens = [24, 48, 96]
    methods = ["Offline", "Online", "FSNet", "OneNet", "Proceed", "ClearE"]
    
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Prediction horizons: {pred_lens}")
    print(f"Methods: {methods}")
    
    total_experiments = len(datasets) * len(models) * len(pred_lens) * len(methods)
    print(f"Total experiments: {total_experiments}")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("results/remaining_experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    results = []
    experiment_count = 0
    successful_count = 0
    failed_count = 0
    
    start_time = time.time()
    
    for dataset in datasets:
        for model in models:
            for pred_len in pred_lens:
                for method in methods:
                    experiment_count += 1
                    print(f"\n=== Experiment {experiment_count}/{total_experiments} ===")
                    
                    result = run_experiment(dataset, model, pred_len, method)
                    results.append(result)
                    
                    if result['success']:
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    print(f"Duration: {result['duration']:.1f}s")
                    
                    # Save intermediate results every 10 experiments
                    if experiment_count % 10 == 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        intermediate_file = results_dir / f"intermediate_results_{timestamp}.json"
                        with open(intermediate_file, 'w') as f:
                            json.dump({
                                'timestamp': timestamp,
                                'total_experiments': total_experiments,
                                'completed_experiments': experiment_count,
                                'successful_experiments': successful_count,
                                'failed_experiments': failed_count,
                                'results': results
                            }, f, indent=2)
                        print(f"Intermediate results saved: {intermediate_file}")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = results_dir / f"final_results_{timestamp}.json"
    
    final_results = {
        'timestamp': timestamp,
        'total_experiments': total_experiments,
        'successful_experiments': successful_count,
        'failed_experiments': failed_count,
        'success_rate': successful_count / total_experiments * 100,
        'total_duration': time.time() - start_time,
        'results': results
    }
    
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n=== Final Results ===")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {successful_count / total_experiments * 100:.1f}%")
    print(f"Total duration: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Results saved: {final_file}")

if __name__ == "__main__":
    main()
