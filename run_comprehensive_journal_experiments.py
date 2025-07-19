#!/usr/bin/env python3
"""
Comprehensive experiment script for journal-quality results
Target: 1,134 experiments with >90% success rate
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

def get_model_hyperparams(model, dataset):
    """Get optimized hyperparameters for each model-dataset combination"""
    
    # Optimized hyperparameters based on dataset characteristics
    hyperparams = {
        # Transformer-based models
        'PatchTST': {
            'default': {'d_model': 16, 'n_heads': 4, 'e_layers': 3, 'd_ff': 128},
            'ECL': {'d_model': 32, 'n_heads': 8, 'e_layers': 3, 'd_ff': 256},
            'gefcom2014': {'d_model': 64, 'n_heads': 8, 'e_layers': 3, 'd_ff': 512},
            'southern_china': {'d_model': 128, 'n_heads': 8, 'e_layers': 3, 'd_ff': 512},
        },
        'iTransformer': {
            'default': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
            'ECL': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256},
            'gefcom2014': {'d_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_ff': 512},
            'southern_china': {'d_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_ff': 512},
        },
        'Autoformer': {'default': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256}},
        'Informer': {'default': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256}},
        'Transformer': {'default': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256}},
        'Crossformer': {'default': {'d_model': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256}},
        
        # Linear models
        'DLinear': {'default': {'d_model': 16, 'n_heads': 4, 'e_layers': 1, 'd_ff': 128}},
        'Linear': {'default': {'d_model': 16, 'n_heads': 4, 'e_layers': 1, 'd_ff': 128}},
        'NLinear': {'default': {'d_model': 16, 'n_heads': 4, 'e_layers': 1, 'd_ff': 128}},
        'RLinear': {'default': {'d_model': 16, 'n_heads': 4, 'e_layers': 1, 'd_ff': 128}},
        
        # CNN and other models
        'TCN': {'default': {'d_model': 32, 'n_heads': 4, 'e_layers': 2, 'd_ff': 128}},
        'TCN_RevIN': {'default': {'d_model': 32, 'n_heads': 4, 'e_layers': 2, 'd_ff': 128}},
        'MTGNN': {'default': {'d_model': 64, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256}},
        'GPT4TS': {'default': {'d_model': 256, 'n_heads': 8, 'e_layers': 3, 'd_ff': 512}},
        'FSNet': {'default': {'d_model': 64, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256}},
        'OneNet': {'default': {'d_model': 64, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256}},
        'LIFT': {'default': {'d_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256}},
        'LightMTS': {'default': {'d_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_ff': 256}},
    }
    
    if model in hyperparams:
        if dataset in hyperparams[model]:
            return hyperparams[model][dataset].values()
        else:
            return hyperparams[model]['default'].values()
    else:
        return 16, 4, 2, 128

def run_single_experiment(dataset, model, pred_len, method, epochs=10, itr=1):
    """Run a single experiment with comprehensive error handling"""
    
    print(f"Running: {dataset} - {model} - {pred_len}h - {method}")
    start_time = time.time()

    # Get model-specific parameters
    d_model, n_heads, e_layers, d_ff = get_model_hyperparams(model, dataset)

    # Base command with optimized parameters
    cmd = [
        "python", "-u", "run.py",
        "--dataset", dataset,
        "--model", model,
        "--seq_len", "96",
        "--pred_len", str(pred_len),
        "--batch_size", "32",
        "--learning_rate", "0.0001",
        "--train_epochs", str(epochs),
        "--itr", str(itr),
        "--features", "M",
        "--d_model", str(d_model),
        "--n_heads", str(n_heads),
        "--e_layers", str(e_layers),
        "--d_ff", str(d_ff),
        "--patience", "3",
        "--des", "Exp"
    ]

    # Add method-specific parameters
    if method == "Offline":
        pass  # Standard offline training
    elif method == "Online":
        cmd.extend([
            "--online_method", "Online",
            "--online_learning_rate", "0.00001",
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
            "--online_learning_rate", "0.00001",
            "--only_test"
        ])
    elif method == "Proceed":
        cmd.extend([
            "--online_method", "Proceed",
            "--online_learning_rate", "0.00001",
            "--only_test"
        ])
    elif method == "ClearE":
        cmd.extend([
            "--online_method", "ClearE",
            "--online_learning_rate", "0.00001",
            "--only_test",
            "--pretrain"
        ])

    try:
        # Run with timeout and comprehensive error handling
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
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
                    'dataset': dataset, 'model': model, 'pred_len': pred_len, 'method': method,
                    'mse': mse, 'mae': mae, 'duration': duration, 'success': True,
                    'hyperparams': {'d_model': d_model, 'n_heads': n_heads, 'e_layers': e_layers, 'd_ff': d_ff}
                }
            else:
                print(f"✗ Failed: Could not parse results")
                return {'dataset': dataset, 'model': model, 'pred_len': pred_len, 'method': method,
                       'success': False, 'error': "Could not parse results", 'duration': duration}
        else:
            print(f"✗ Failed: {result.stderr[:200]}...")
            return {'dataset': dataset, 'model': model, 'pred_len': pred_len, 'method': method,
                   'success': False, 'error': result.stderr, 'duration': duration}
            
    except subprocess.TimeoutExpired:
        print(f"✗ Failed: Timeout after 2 hours")
        return {'dataset': dataset, 'model': model, 'pred_len': pred_len, 'method': method,
               'success': False, 'error': "Timeout after 2 hours", 'duration': 7200}
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return {'dataset': dataset, 'model': model, 'pred_len': pred_len, 'method': method,
               'success': False, 'error': str(e), 'duration': time.time() - start_time}

def main():
    """Main comprehensive experiment execution"""
    print("=== COMPREHENSIVE JOURNAL EXPERIMENTS ===")
    print("Target: 1,134 experiments with >90% success rate")
    
    # Complete experimental configuration
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "gefcom2014", "southern_china"]
    models = [
        "PatchTST", "iTransformer", "Autoformer", "Informer", "Transformer", "Crossformer",
        "DLinear", "Linear", "NLinear", "RLinear", "TCN", "TCN_RevIN", "MTGNN",
        "GPT4TS", "FSNet", "OneNet", "LIFT", "LightMTS"
    ]
    pred_lens = [24, 48, 96]
    methods = ["Offline", "Online", "FSNet", "OneNet", "Proceed", "ClearE"]
    
    total_experiments = len(datasets) * len(models) * len(pred_lens) * len(methods)
    print(f"Total planned experiments: {total_experiments}")
    
    # Create results directory
    results_dir = Path("results/comprehensive_journal_experiments")
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
                    
                    result = run_single_experiment(dataset, model, pred_len, method)
                    results.append(result)
                    
                    if result['success']:
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    success_rate = successful_count / experiment_count * 100
                    print(f"Progress: {experiment_count}/{total_experiments} ({success_rate:.1f}% success)")
                    
                    # Save intermediate results every 50 experiments
                    if experiment_count % 50 == 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        intermediate_file = results_dir / f"intermediate_results_{timestamp}.json"
                        with open(intermediate_file, 'w') as f:
                            json.dump({
                                'timestamp': timestamp,
                                'total_experiments': total_experiments,
                                'completed_experiments': experiment_count,
                                'successful_experiments': successful_count,
                                'failed_experiments': failed_count,
                                'success_rate': success_rate,
                                'results': results
                            }, f, indent=2)
                        print(f"Intermediate results saved: {intermediate_file}")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = results_dir / f"final_comprehensive_results_{timestamp}.json"
    
    final_results = {
        'timestamp': timestamp,
        'total_experiments': total_experiments,
        'successful_experiments': successful_count,
        'failed_experiments': failed_count,
        'success_rate': successful_count / total_experiments * 100,
        'total_duration': time.time() - start_time,
        'datasets': datasets,
        'models': models,
        'methods': methods,
        'prediction_horizons': pred_lens,
        'results': results
    }
    
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {successful_count / total_experiments * 100:.1f}%")
    print(f"Total duration: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Results saved: {final_file}")
    
    # Create summary for journal paper
    if successful_count / total_experiments >= 0.9:
        print("\n🎉 SUCCESS: >90% success rate achieved - ready for journal submission!")
    else:
        print(f"\n⚠️  WARNING: {successful_count / total_experiments * 100:.1f}% success rate - may need additional fixes")

if __name__ == "__main__":
    main()
