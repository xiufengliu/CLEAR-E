#!/usr/bin/env python3
"""
Run Comprehensive CLEAR-E Experiments for Paper Results
"""

import os
import subprocess
import json
import time
from datetime import datetime

def get_model_hyperparams(model, dataset):
    """Get model-specific hyperparameters based on the model and dataset"""

    # Default parameters
    d_model, n_heads, e_layers, d_ff = 512, 8, 2, 2048

    if model == "PatchTST":
        e_layers = 3
        if dataset in ['ETTh1', 'ETTh2']:
            d_model, n_heads, d_ff = 16, 4, 128
        elif dataset in ['ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']:
            d_model, n_heads, d_ff = 128, 16, 256
        else:
            d_model, n_heads, d_ff = 64, 16, 128

    elif model == "iTransformer":
        e_layers = 3
        d_model, d_ff = 512, 512
        if dataset == 'Traffic':
            e_layers = 4
        elif 'ETT' in dataset:
            e_layers = 2
            if dataset == 'ETTh1':
                d_model, d_ff = 256, 256
            else:
                d_model, d_ff = 128, 128

    elif model == "Crossformer":
        e_layers, d_ff, d_model, n_heads = 3, 512, 256, 4
        if dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']:
            d_model, n_heads = 256, 4
        else:
            d_model, n_heads = 64, 2
        if dataset in ['Traffic', 'ECL']:
            d_ff = 128
        if dataset in ['Illness']:
            e_layers = 2

    elif model == "GPT4TS":
        e_layers, d_model, n_heads, d_ff = 3, 768, 4, 768

    elif model == "Autoformer":
        e_layers = 2
        d_model, n_heads, d_ff = 512, 8, 2048

    elif model == "Informer":
        e_layers = 2
        d_model, n_heads, d_ff = 512, 8, 2048

    elif model == "Transformer":
        e_layers = 2
        d_model, n_heads, d_ff = 512, 8, 2048

    elif model in ["DLinear", "Linear", "NLinear", "RLinear"]:
        # Linear models don't use these parameters, but we set defaults
        d_model, n_heads, e_layers, d_ff = 512, 8, 2, 2048

    elif model in ["TCN", "TCN_RevIN"]:
        # TCN models use different architecture
        d_model, n_heads, e_layers, d_ff = 512, 8, 2, 2048

    elif model == "MTGNN":
        # Graph neural network
        d_model, n_heads, e_layers, d_ff = 512, 8, 2, 2048

    elif model in ["FSNet", "OneNet", "LIFT", "LightMTS"]:
        # Other advanced models
        d_model, n_heads, e_layers, d_ff = 512, 8, 2, 2048

    return d_model, n_heads, e_layers, d_ff

def run_experiment(dataset, model, pred_len, method, epochs=5, itr=1):
    """Run a single experiment and return results"""

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
    if method == "ClearE":
        cmd.extend([
            "--online_learning_rate", "0.0001",
            "--online_method", "ClearE",
            "--concept_dim", "64",
            "--bottleneck_dim", "32",
            "--metadata_dim", "10",
            "--metadata_hidden_dim", "32",
            "--drift_memory_size", "10",
            "--drift_reg_weight", "0.1",
            "--use_energy_loss",
            "--high_load_threshold", "0.8",
            "--underestimate_penalty", "2.0",
            "--border_type", "online",
            "--pretrain",
            "--save_opt",
            "--only_test",
            "--val_online_lr",
            "--diff_online_lr",
            "--tune_mode", "down_up"
        ])

        # Apply the same checkpoint path fix for ClearE
        import os
        import shutil

        # Get model-specific parameters for checkpoint path
        d_model, n_heads, e_layers, d_ff = get_model_hyperparams(model, dataset)
        source_path = f"./checkpoints/{dataset}_96_{pred_len}_{model}_online_ftM_sl96_ll48_pl{pred_len}_lr0.001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0"
        target_path = f"./checkpoints/{dataset}_96_{pred_len}_{model}_online_ftM_sl96_ll48_pl{pred_len}_lr0.0001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0"

        # Create target directory and copy checkpoint if source exists
        os.makedirs(target_path, exist_ok=True)
        source_checkpoint = f"{source_path}/checkpoint.pth"
        target_checkpoint = f"{target_path}/checkpoint.pth"

        if os.path.exists(source_checkpoint) and not os.path.exists(target_checkpoint):
            print(f"Copying checkpoint from {source_checkpoint} to {target_checkpoint}")
            try:
                import shutil
                shutil.copy2(source_checkpoint, target_checkpoint)
                print(f"Successfully copied checkpoint")
            except Exception as e:
                print(f"Error copying checkpoint: {e}")
        elif not os.path.exists(source_checkpoint):
            print(f"Warning: Source checkpoint not found: {source_checkpoint}")
        else:
            print(f"Target checkpoint already exists: {target_checkpoint}")
    elif method == "Online":
        cmd.extend([
            "--online_learning_rate", "0.0001",
            "--online_method", "Online",
            "--border_type", "online",
            "--pretrain",
            "--save_opt",
            "--only_test"
        ])

        # Fix the checkpoint path issue by copying from the 0.001 checkpoint to the 0.0001 checkpoint
        # This is a workaround for the mismatch between the learning rate in the checkpoint path
        # and the actual learning rate used for training
        import os
        import shutil

        # Get model-specific parameters for checkpoint path
        d_model, n_heads, e_layers, d_ff = get_model_hyperparams(model, dataset)
        source_path = f"./checkpoints/{dataset}_96_{pred_len}_{model}_online_ftM_sl96_ll48_pl{pred_len}_lr0.001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0"
        target_path = f"./checkpoints/{dataset}_96_{pred_len}_{model}_online_ftM_sl96_ll48_pl{pred_len}_lr0.0001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0"

        # Create target directory and copy checkpoint if source exists
        os.makedirs(target_path, exist_ok=True)
        source_checkpoint = f"{source_path}/checkpoint.pth"
        target_checkpoint = f"{target_path}/checkpoint.pth"

        if os.path.exists(source_checkpoint) and not os.path.exists(target_checkpoint):
            print(f"Copying checkpoint from {source_checkpoint} to {target_checkpoint}")
            try:
                shutil.copy2(source_checkpoint, target_checkpoint)
                print(f"Successfully copied checkpoint")
            except Exception as e:
                print(f"Error copying checkpoint: {e}")
        elif not os.path.exists(source_checkpoint):
            print(f"Warning: Source checkpoint not found: {source_checkpoint}")
        else:
            print(f"Target checkpoint already exists: {target_checkpoint}")
    elif method == "Proceed":
        cmd.extend([
            "--online_learning_rate", "0.0001",
            "--online_method", "Proceed",
            "--concept_dim", "64",
            "--bottleneck_dim", "32",
            "--border_type", "online",
            "--pretrain",
            "--save_opt",
            "--only_test",
            "--val_online_lr",
            "--diff_online_lr",
            "--tune_mode", "down_up"
        ])

        # Apply the same checkpoint path fix for Proceed
        import os
        d_model, n_heads, e_layers, d_ff = get_model_hyperparams(model, dataset)
        source_path = f"./checkpoints/{dataset}_96_{pred_len}_{model}_online_ftM_sl96_ll48_pl{pred_len}_lr0.001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0"
        target_path = f"./checkpoints/{dataset}_96_{pred_len}_{model}_online_ftM_sl96_ll48_pl{pred_len}_lr0.0001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0"

        os.makedirs(target_path, exist_ok=True)
        if os.path.exists(f"{source_path}/checkpoint.pth") and not os.path.exists(f"{target_path}/checkpoint.pth"):
            try:
                import shutil
                shutil.copy2(f"{source_path}/checkpoint.pth", f"{target_path}/checkpoint.pth")
                print(f"Successfully copied checkpoint for Proceed method")
            except Exception as e:
                print(f"Error copying checkpoint for Proceed method: {e}")
    elif method == "FSNet":
        # For FSNet, we need to handle the model differently
        if model == "TCN_RevIN":
            # Use FSNet directly for TCN_RevIN
            cmd[4] = "FSNet"  # Replace model parameter
            cmd.extend([
                "--online_learning_rate", "0.00003",
                "--normalization", "RevIN",
                "--border_type", "online",
                "--save_opt",
                "--only_test"
            ])
        else:
            # For other models, use FSNet wrapper
            cmd.extend([
                "--online_learning_rate", "0.00003",
                "--online_method", "FSNet",
                "--border_type", "online",
                "--save_opt",
                "--only_test"
            ])
    elif method == "OneNet":
        cmd.extend([
            "--online_learning_rate", "0.0001",
            "--online_method", "OneNet",
            "--border_type", "online",
            "--save_opt",
            "--only_test"
        ])

        # Apply the same checkpoint path fix for OneNet
        import os
        d_model, n_heads, e_layers, d_ff = get_model_hyperparams(model, dataset)
        source_path = f"./checkpoints/{dataset}_96_{pred_len}_{model}_online_ftM_sl96_ll48_pl{pred_len}_lr0.001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0"
        target_path = f"./checkpoints/{dataset}_96_{pred_len}_{model}_online_ftM_sl96_ll48_pl{pred_len}_lr0.0001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0"

        os.makedirs(target_path, exist_ok=True)
        if os.path.exists(f"{source_path}/checkpoint.pth") and not os.path.exists(f"{target_path}/checkpoint.pth"):
            try:
                import shutil
                shutil.copy2(f"{source_path}/checkpoint.pth", f"{target_path}/checkpoint.pth")
                print(f"Successfully copied checkpoint for OneNet method")
            except Exception as e:
                print(f"Error copying checkpoint for OneNet method: {e}")
    elif method == "SOLID++":
        cmd.extend([
            "--online_learning_rate", "0.0001",
            "--online_method", "SOLID",
            "--border_type", "online",
            "--save_opt",
            "--only_test"
        ])

        # Apply the same checkpoint path fix for SOLID++
        import os
        d_model, n_heads, e_layers, d_ff = get_model_hyperparams(model, dataset)
        source_path = f"./checkpoints/{dataset}_96_{pred_len}_{model}_online_ftM_sl96_ll48_pl{pred_len}_lr0.001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0"
        target_path = f"./checkpoints/{dataset}_96_{pred_len}_{model}_online_ftM_sl96_ll48_pl{pred_len}_lr0.0001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0"

        os.makedirs(target_path, exist_ok=True)
        if os.path.exists(f"{source_path}/checkpoint.pth") and not os.path.exists(f"{target_path}/checkpoint.pth"):
            try:
                import shutil
                shutil.copy2(f"{source_path}/checkpoint.pth", f"{target_path}/checkpoint.pth")
                print(f"Successfully copied checkpoint for SOLID++ method")
            except Exception as e:
                print(f"Error copying checkpoint for SOLID++ method: {e}")
    else:
        # Offline baseline - just train and test
        pass
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            # Extract results from output
            output = result.stdout
            lines = output.split('\n')
            
            for line in lines:
                if line.startswith('mse:') and 'mae:' in line:
                    parts = line.split(',')
                    mse = float(parts[0].split(':')[1].strip())
                    mae = float(parts[1].split(':')[1].strip())
                    
                    duration = time.time() - start_time
                    return {
                        'dataset': dataset,
                        'model': model,
                        'pred_len': pred_len,
                        'method': method,
                        'mse': mse,
                        'mae': mae,
                        'duration': duration,
                        'success': True,
                        'output': output
                    }
            
            # If no results found in output
            return {
                'dataset': dataset,
                'model': model,
                'pred_len': pred_len,
                'method': method,
                'success': False,
                'error': 'No results found in output',
                'output': output,
                'duration': time.time() - start_time
            }
        else:
            return {
                'dataset': dataset,
                'model': model,
                'pred_len': pred_len,
                'method': method,
                'success': False,
                'error': result.stderr,
                'output': result.stdout,
                'duration': time.time() - start_time
            }
            
    except subprocess.TimeoutExpired:
        return {
            'dataset': dataset,
            'model': model,
            'pred_len': pred_len,
            'method': method,
            'success': False,
            'error': 'Timeout',
            'duration': time.time() - start_time
        }
    except Exception as e:
        return {
            'dataset': dataset,
            'model': model,
            'pred_len': pred_len,
            'method': method,
            'success': False,
            'error': str(e),
            'duration': time.time() - start_time
        }

def ensure_model_trained(dataset, model, epochs=5):
    """Ensure the base model is trained before running online experiments"""

    # Check for existing checkpoints with different learning rates and model parameters
    # Generate checkpoint path based on model hyperparameters
    d_model, n_heads, e_layers, d_ff = get_model_hyperparams(model, dataset)

    possible_checkpoints = [
        f"./checkpoints/{dataset}_96_24_{model}_online_ftM_sl96_ll48_pl24_lr0.001_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth"
    ]

    for checkpoint_path in possible_checkpoints:
        if os.path.exists(checkpoint_path):
            print(f"✓ Model {model} already trained for {dataset}")
            print(f"  Found checkpoint: {checkpoint_path}")
            return True

    print(f"Training {model} for {dataset}...")

    # Set model-specific parameters based on dataset and model using hyperparameter configurations
    d_model, n_heads, e_layers, d_ff = get_model_hyperparams(model, dataset)

    cmd = [
        "python", "-u", "run.py",
        "--dataset", dataset,
        "--model", model,
        "--seq_len", "96",
        "--pred_len", "24",
        "--batch_size", "16",
        "--learning_rate", "0.001",
        "--train_epochs", str(epochs),
        "--itr", "1",
        "--features", "M",
        "--d_model", str(d_model),
        "--n_heads", str(n_heads),
        "--e_layers", str(e_layers),
        "--d_ff", str(d_ff)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            print(f"✓ Successfully trained {model} for {dataset}")
            return True
        else:
            print(f"✗ Failed to train {model} for {dataset}: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error training {model} for {dataset}: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CLEAR-E Comprehensive Experiments for TNNLS')
    parser.add_argument('--mode', choices=['full', 'quick', 'test'], default='quick',
                       help='Experiment mode: full (all models), quick (subset), test (minimal)')
    parser.add_argument('--models', nargs='+',
                       help='Specific models to test (overrides mode selection)')
    parser.add_argument('--methods', nargs='+',
                       help='Specific methods to test (overrides default)')
    args = parser.parse_args()

    # Comprehensive experimental configuration for TNNLS journal
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "gefcom2014", "southern_china"]

    # Model selection based on mode
    if args.models:
        models = args.models
    elif args.mode == 'full':
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
    elif args.mode == 'quick':
        # Subset of key models for faster testing
        models = [
            "PatchTST", "iTransformer", "DLinear", "Autoformer", "Informer",
            "Transformer", "TCN_RevIN", "Linear", "NLinear"
        ]
    else:  # test mode
        # Minimal set for testing
        models = ["PatchTST", "DLinear"]

    pred_lens = [24, 48, 96]

    # Method selection
    if args.methods:
        methods = args.methods
    else:
        # Comprehensive online learning methods for TNNLS journal
        methods = [
            "Offline",      # Baseline offline training
            "Online",       # Simple online fine-tuning
            "FSNet",        # Few-shot network adaptation
            "OneNet",       # One-shot online learning
            "Proceed",      # Progressive concept drift adaptation
            "ClearE"        # Our proposed method
        ]

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs/comprehensive_experiments"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/experiment_log_{timestamp}.txt"

    # Create a function to log messages to both console and file
    def log_message(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    log_message("=== Comprehensive CLEAR-E Experiments ===")
    log_message(f"Datasets: {datasets}")
    log_message(f"Models: {models}")
    log_message(f"Prediction horizons: {pred_lens}")
    log_message(f"Methods: {methods}")
    log_message("==========================================")

    # Create results directory
    os.makedirs("results/comprehensive_experiments", exist_ok=True)

    all_results = []
    total_experiments = len(datasets) * len(models) * len(pred_lens) * len(methods)
    current_experiment = 0

    # Track successful and failed experiments
    success_count = 0
    failure_count = 0
    skipped_count = 0

    # Create a progress tracking file that can be monitored
    progress_file = f"{log_dir}/progress_{timestamp}.txt"

    for dataset in datasets:
        for model in models:
            # Ensure base model is trained
            if not ensure_model_trained(dataset, model):
                log_message(f"Skipping {dataset}-{model} due to training failure")
                skipped_count += len(pred_lens) * len(methods)
                continue

            for pred_len in pred_lens:
                for method in methods:
                    current_experiment += 1
                    log_message(f"\n=== Experiment {current_experiment}/{total_experiments} ===")

                    # Update progress file
                    with open(progress_file, "w") as f:
                        f.write(f"Progress: {current_experiment}/{total_experiments}\n")
                        f.write(f"Success: {success_count}, Failed: {failure_count}, Skipped: {skipped_count}\n")
                        f.write(f"Current: {dataset} - {model} - {pred_len}h - {method}\n")

                    try:
                        result = run_experiment(dataset, model, pred_len, method)
                        all_results.append(result)

                        if result['success']:
                            log_message(f"✓ Success: MSE={result['mse']:.4f}, MAE={result['mae']:.4f}")
                            success_count += 1
                        else:
                            log_message(f"✗ Failed: {result.get('error', 'Unknown error')}")
                            failure_count += 1

                        log_message(f"Duration: {result['duration']:.1f}s")
                    except Exception as e:
                        log_message(f"✗ Exception during experiment: {str(e)}")
                        failure_count += 1
                        all_results.append({
                            'dataset': dataset,
                            'model': model,
                            'pred_len': pred_len,
                            'method': method,
                            'success': False,
                            'error': str(e),
                            'duration': 0
                        })

    # Save results
    results_file = f"results/comprehensive_experiments/results_{timestamp}.json"

    summary = {
        'timestamp': timestamp,
        'total_experiments': total_experiments,
        'successful_experiments': success_count,
        'failed_experiments': failure_count,
        'skipped_experiments': skipped_count,
        'results': all_results
    }

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    log_message(f"\n=== Final Experiment Summary ===")
    log_message(f"Total experiments planned: {total_experiments}")
    log_message(f"Successful: {success_count}")
    log_message(f"Failed: {failure_count}")
    log_message(f"Skipped: {skipped_count}")
    log_message(f"Success rate: {success_count/(total_experiments-skipped_count)*100:.1f}% (of attempted)")
    log_message(f"Results saved to: {results_file}")
    log_message(f"Log saved to: {log_file}")

    # Create summary table
    create_summary_table(all_results, timestamp)

    # Update final progress
    with open(progress_file, "w") as f:
        f.write("COMPLETED\n")
        f.write(f"Final Results: Success: {success_count}, Failed: {failure_count}, Skipped: {skipped_count}\n")
        f.write(f"Success rate: {success_count/(total_experiments-skipped_count)*100:.1f}%\n")

def create_summary_table(results, timestamp):
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
        
        summary_data[key]['mse_values'].append(result['mse'])
        summary_data[key]['mae_values'].append(result['mae'])
    
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
    table_file = f"results/comprehensive_experiments/summary_table_{timestamp}.json"
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
