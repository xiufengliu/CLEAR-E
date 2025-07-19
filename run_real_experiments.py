#!/usr/bin/env python3
"""
Run Real CLEAR-E Experiments
This script runs comprehensive experiments to generate real results for the paper
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

def run_command(cmd, log_file=None):
    """Run a command and capture output"""
    print(f"Running: {' '.join(cmd)}")
    
    if log_file:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            process.wait()
            return process.returncode
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode

def extract_results_from_log(log_file):
    """Extract MSE and MAE results from log file"""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Look for result patterns
        lines = content.split('\n')
        for line in lines:
            if line.startswith('mse:') and 'mae:' in line:
                parts = line.split(',')
                mse = float(parts[0].split(':')[1].strip())
                mae = float(parts[1].split(':')[1].strip())
                return {'mse': mse, 'mae': mae}
                
        # Alternative pattern search
        for line in lines:
            if 'MSE:' in line and 'MAE:' in line:
                # Extract numbers from line
                import re
                numbers = re.findall(r'[\d.]+', line)
                if len(numbers) >= 2:
                    return {'mse': float(numbers[0]), 'mae': float(numbers[1])}
                    
    except Exception as e:
        print(f"Error extracting results from {log_file}: {e}")
    
    return None

def run_single_experiment(dataset, model, pred_len, method, config):
    """Run a single experiment configuration"""
    
    # Create log file name
    log_file = f"logs/{model}_{method}_{dataset}_{pred_len}.log"
    os.makedirs("logs", exist_ok=True)
    
    # Base command
    cmd = [
        "python", "-u", "run.py",
        "--dataset", dataset,
        "--model", model,
        "--seq_len", str(config['seq_len']),
        "--pred_len", str(pred_len),
        "--batch_size", str(config['batch_size']),
        "--learning_rate", str(config['learning_rate']),
        "--online_learning_rate", str(config['online_learning_rate']),
        "--train_epochs", str(config['epochs']),
        "--itr", str(config['itr']),
        "--border_type", "online",
        "--pretrain",
        "--save_opt",
        "--only_test",
        "--features", "M"
    ]
    
    # Add method-specific parameters
    if method == "ClearE":
        cmd.extend([
            "--online_method", "ClearE",
            "--concept_dim", str(config['concept_dim']),
            "--bottleneck_dim", str(config['bottleneck_dim']),
            "--metadata_dim", str(config['metadata_dim']),
            "--metadata_hidden_dim", str(config['metadata_hidden_dim']),
            "--drift_memory_size", str(config['drift_memory_size']),
            "--drift_reg_weight", str(config['drift_reg_weight']),
            "--use_energy_loss",
            "--high_load_threshold", str(config['high_load_threshold']),
            "--underestimate_penalty", str(config['underestimate_penalty']),
            "--val_online_lr",
            "--diff_online_lr",
            "--tune_mode", "down_up"
        ])
    elif method == "Proceed":
        cmd.extend([
            "--online_method", "Proceed",
            "--concept_dim", str(config['concept_dim']),
            "--bottleneck_dim", str(config['bottleneck_dim']),
            "--val_online_lr",
            "--diff_online_lr",
            "--tune_mode", "down_up"
        ])
    else:
        cmd.extend([
            "--online_method", method
        ])
    
    # Run the experiment
    print(f"Running: {dataset} - {model} - {pred_len}h - {method}")
    start_time = time.time()
    
    return_code = run_command(cmd, log_file)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Extract results
    results = extract_results_from_log(log_file)
    
    return {
        'dataset': dataset,
        'model': model,
        'pred_len': pred_len,
        'method': method,
        'return_code': return_code,
        'duration': duration,
        'log_file': log_file,
        'results': results,
        'success': return_code == 0 and results is not None
    }

def main():
    parser = argparse.ArgumentParser(description="Run CLEAR-E Real Experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick test with reduced parameters")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL"], 
                       help="Datasets to test")
    parser.add_argument("--models", nargs="+", default=["PatchTST", "DLinear"], 
                       help="Models to test")
    parser.add_argument("--pred_lens", nargs="+", type=int, default=[24, 48, 96], 
                       help="Prediction horizons")
    parser.add_argument("--methods", nargs="+", default=["ClearE", "Proceed", "Online"], 
                       help="Methods to test")
    
    args = parser.parse_args()
    
    # Configuration
    if args.quick:
        config = {
            'seq_len': 96,
            'batch_size': 16,
            'learning_rate': 0.001,
            'online_learning_rate': 0.0001,
            'epochs': 5,
            'itr': 1,
            'concept_dim': 32,
            'bottleneck_dim': 16,
            'metadata_dim': 10,
            'metadata_hidden_dim': 16,
            'drift_memory_size': 5,
            'drift_reg_weight': 0.1,
            'high_load_threshold': 0.8,
            'underestimate_penalty': 2.0
        }
    else:
        config = {
            'seq_len': 96,
            'batch_size': 32,
            'learning_rate': 0.001,
            'online_learning_rate': 0.0001,
            'epochs': 50,
            'itr': 3,
            'concept_dim': 64,
            'bottleneck_dim': 32,
            'metadata_dim': 10,
            'metadata_hidden_dim': 32,
            'drift_memory_size': 10,
            'drift_reg_weight': 0.1,
            'high_load_threshold': 0.8,
            'underestimate_penalty': 2.0
        }
    
    print("=== CLEAR-E Real Experiments ===")
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Prediction horizons: {args.pred_lens}")
    print(f"Methods: {args.methods}")
    print(f"Quick mode: {args.quick}")
    print("================================")
    
    # Create results directory
    results_dir = "results/real_experiments"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments
    all_results = []
    total_experiments = len(args.datasets) * len(args.models) * len(args.pred_lens) * len(args.methods)
    current_experiment = 0
    
    for dataset in args.datasets:
        for model in args.models:
            for pred_len in args.pred_lens:
                for method in args.methods:
                    current_experiment += 1
                    print(f"\n=== Experiment {current_experiment}/{total_experiments} ===")
                    
                    result = run_single_experiment(dataset, model, pred_len, method, config)
                    all_results.append(result)
                    
                    if result['success']:
                        print(f"✓ Success: MSE={result['results']['mse']:.4f}, MAE={result['results']['mae']:.4f}")
                    else:
                        print(f"✗ Failed: {result['log_file']}")
                    
                    print(f"Duration: {result['duration']:.1f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{results_dir}/experiment_results_{timestamp}.json"
    
    summary = {
        'timestamp': timestamp,
        'config': config,
        'args': vars(args),
        'total_experiments': total_experiments,
        'successful_experiments': sum(1 for r in all_results if r['success']),
        'results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Experiment Summary ===")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {summary['successful_experiments']}")
    print(f"Failed: {total_experiments - summary['successful_experiments']}")
    print(f"Success rate: {summary['successful_experiments']/total_experiments*100:.1f}%")
    print(f"Results saved to: {results_file}")
    
    # Create summary table
    create_summary_table(all_results, results_dir, timestamp)

def create_summary_table(results, results_dir, timestamp):
    """Create a summary table of results"""
    
    # Group results by dataset and method
    summary_data = {}
    
    for result in results:
        if not result['success']:
            continue
            
        key = f"{result['dataset']}_{result['method']}"
        if key not in summary_data:
            summary_data[key] = {
                'dataset': result['dataset'],
                'method': result['method'],
                'mse_values': [],
                'mae_values': []
            }
        
        summary_data[key]['mse_values'].append(result['results']['mse'])
        summary_data[key]['mae_values'].append(result['results']['mae'])
    
    # Calculate statistics
    import numpy as np
    
    table_data = []
    for key, data in summary_data.items():
        if data['mse_values']:
            mse_mean = np.mean(data['mse_values'])
            mse_std = np.std(data['mse_values'])
            mae_mean = np.mean(data['mae_values'])
            mae_std = np.std(data['mae_values'])
            
            table_data.append({
                'dataset': data['dataset'],
                'method': data['method'],
                'mse_mean': mse_mean,
                'mse_std': mse_std,
                'mae_mean': mae_mean,
                'mae_std': mae_std,
                'n_results': len(data['mse_values'])
            })
    
    # Save table
    table_file = f"{results_dir}/summary_table_{timestamp}.json"
    with open(table_file, 'w') as f:
        json.dump(table_data, f, indent=2)
    
    print(f"Summary table saved to: {table_file}")
    
    # Print table
    print("\n=== Results Summary Table ===")
    print(f"{'Dataset':<15} {'Method':<10} {'MSE':<15} {'MAE':<15} {'N':<5}")
    print("-" * 65)
    
    for row in sorted(table_data, key=lambda x: (x['dataset'], x['method'])):
        mse_str = f"{row['mse_mean']:.4f}±{row['mse_std']:.4f}"
        mae_str = f"{row['mae_mean']:.4f}±{row['mae_std']:.4f}"
        print(f"{row['dataset']:<15} {row['method']:<10} {mse_str:<15} {mae_str:<15} {row['n_results']:<5}")

if __name__ == "__main__":
    main()
