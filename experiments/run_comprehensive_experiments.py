#!/usr/bin/env python3
"""
Comprehensive Online Learning Experiments for CLEAR-E
Following the experimental protocol from PROCEED paper with multiple horizons and methods
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ComprehensiveExperimentRunner:
    def __init__(self):
        self.base_models = ['TCN_RevIN', 'PatchTST', 'iTransformer']
        self.online_methods = ['Online', 'FSNet', 'OneNet', 'SOLID++', 'Proceed', 'ClearE']
        self.datasets = ['ETTh2', 'ETTm1', 'Weather', 'ECL', 'Traffic']
        self.horizons = [24, 48, 96]
        self.results = {}
        
        # Create results directory
        self.results_dir = Path("results/comprehensive_experiments")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_single_experiment(self, model, method, dataset, horizon):
        """Run a single experiment configuration"""
        print(f"Running: {model} + {method} on {dataset} with horizon {horizon}")

        # Construct script path - need to go up one directory from experiments
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if method == 'ClearE':
            script_path = os.path.join(base_path, f"scripts/online/{model}/ClearE/{dataset}.sh")
        else:
            script_path = os.path.join(base_path, f"scripts/online/{model}/{method}/{dataset}.sh")

        if not os.path.exists(script_path):
            print(f"Script not found: {script_path}")
            return None
            
        try:
            # Modify the script to use the specified horizon
            self._modify_script_horizon(script_path, horizon)

            # Run the experiment from project root directory
            result = subprocess.run(
                ['bash', script_path],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=base_path  # Run from project root
            )
            
            if result.returncode == 0:
                # Parse results from output
                metrics = self._parse_experiment_output(result.stdout)
                return metrics
            else:
                print(f"Experiment failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Experiment timed out")
            return None
        except Exception as e:
            print(f"Error running experiment: {e}")
            return None
    
    def _modify_script_horizon(self, script_path, horizon):
        """Modify script to use specified prediction horizon"""
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Replace pred_len parameter
        content = content.replace('pred_len in 24 48 96', f'pred_len in {horizon}')
        content = content.replace('for pred_len in 24 48 96', f'for pred_len in {horizon}')
        
        with open(script_path, 'w') as f:
            f.write(content)
    
    def _parse_experiment_output(self, output):
        """Parse experiment output to extract metrics"""
        lines = output.split('\n')
        metrics = {}
        
        for line in lines:
            if 'mse:' in line.lower():
                try:
                    mse_value = float(line.split('mse:')[1].split()[0])
                    metrics['mse'] = mse_value
                except:
                    pass
            if 'mae:' in line.lower():
                try:
                    mae_value = float(line.split('mae:')[1].split()[0])
                    metrics['mae'] = mae_value
                except:
                    pass
        
        return metrics if metrics else None
    
    def run_all_experiments(self):
        """Run all experiment combinations"""
        total_experiments = len(self.base_models) * len(self.online_methods) * len(self.datasets) * len(self.horizons)
        current_experiment = 0
        
        print(f"Starting {total_experiments} experiments...")
        start_time = time.time()
        
        for model in self.base_models:
            for method in self.online_methods:
                for dataset in self.datasets:
                    for horizon in self.horizons:
                        current_experiment += 1
                        print(f"\nExperiment {current_experiment}/{total_experiments}")
                        
                        # Skip if method not available for model
                        if not self._method_available(model, method, dataset):
                            print(f"Skipping {model}+{method} on {dataset} (not available)")
                            continue
                        
                        result = self.run_single_experiment(model, method, dataset, horizon)
                        
                        if result:
                            key = f"{model}_{method}_{dataset}_{horizon}"
                            self.results[key] = result
                            print(f"✓ MSE: {result.get('mse', 'N/A'):.3f}, MAE: {result.get('mae', 'N/A'):.3f}")
                        else:
                            print("✗ Experiment failed")
                        
                        # Save intermediate results
                        if current_experiment % 10 == 0:
                            self._save_intermediate_results()
        
        elapsed_time = time.time() - start_time
        print(f"\nAll experiments completed in {elapsed_time/3600:.2f} hours")
        
        # Save final results
        self._save_final_results()
        self._generate_tables()
    
    def _method_available(self, model, method, dataset):
        """Check if method is available for model and dataset"""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if method == 'ClearE':
            script_path = os.path.join(base_path, f"scripts/online/{model}/ClearE/{dataset}.sh")
        else:
            script_path = os.path.join(base_path, f"scripts/online/{model}/{method}/{dataset}.sh")
        return os.path.exists(script_path)
    
    def _save_intermediate_results(self):
        """Save intermediate results"""
        results_file = self.results_dir / "intermediate_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _save_final_results(self):
        """Save final results in multiple formats"""
        # JSON format
        json_file = self.results_dir / "final_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # CSV format
        csv_data = []
        for key, metrics in self.results.items():
            parts = key.split('_')
            model = parts[0]
            method = parts[1]
            dataset = parts[2]
            horizon = parts[3]
            
            row = {
                'Model': model,
                'Method': method,
                'Dataset': dataset,
                'Horizon': horizon,
                'MSE': metrics.get('mse', np.nan),
                'MAE': metrics.get('mae', np.nan)
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = self.results_dir / "final_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")
    
    def _generate_tables(self):
        """Generate LaTeX tables in the format shown"""
        self._generate_mse_table()
        self._generate_mae_table()
    
    def _generate_mse_table(self):
        """Generate MSE table in LaTeX format"""
        print("\nGenerating MSE table...")
        
        # Create table structure
        table_lines = []
        table_lines.append("\\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|}")
        table_lines.append("\\hline \\multirow[b]{2}{*}{Model} & \\multirow{2}{*}{\\begin{tabular}{l}")
        table_lines.append("Dataset \\\\")
        table_lines.append("Method")
        table_lines.append("\\end{tabular}} & \\multicolumn{3}{|c|}{ETTh2} & \\multicolumn{3}{|c|}{ETTm1} & \\multicolumn{3}{|c|}{Weather} & \\multicolumn{3}{|c|}{ECL} & \\multicolumn{3}{|c|}{Traffic} \\\\")
        table_lines.append("\\hline & & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 \\\\")
        
        # Add data rows
        for model in self.base_models:
            model_name = model.replace('_RevIN', '')  # Clean model name
            first_row = True
            
            for method in self.online_methods:
                if first_row:
                    row = f"\\multirow{{{len(self.online_methods)}}}{{*}}{{{model_name}}} & {method}"
                    first_row = False
                else:
                    row = f" & {method}"
                
                for dataset in self.datasets:
                    for horizon in self.horizons:
                        key = f"{model}_{method}_{dataset}_{horizon}"
                        if key in self.results and 'mse' in self.results[key]:
                            mse_val = self.results[key]['mse']
                            row += f" & {mse_val:.3f}"
                        else:
                            row += " & -"
                
                row += " \\\\"
                table_lines.append(row)
        
        table_lines.append("\\end{tabular}")
        
        # Save table
        table_file = self.results_dir / "mse_table.tex"
        with open(table_file, 'w') as f:
            f.write('\n'.join(table_lines))
        
        print(f"MSE table saved to {table_file}")

    def _generate_mae_table(self):
        """Generate MAE table in LaTeX format"""
        print("\nGenerating MAE table...")

        # Create table structure
        table_lines = []
        table_lines.append("\\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|}")
        table_lines.append("\\hline \\multirow[b]{2}{*}{Model} & Dataset & \\multicolumn{3}{|c|}{ETTh2} & \\multicolumn{3}{|c|}{ETTm1} & \\multicolumn{3}{|c|}{Weather} & \\multicolumn{3}{|c|}{ECL} & \\multicolumn{3}{|c|}{Traffic} \\\\")
        table_lines.append("\\hline & & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 \\\\")

        # Add data rows
        for model in self.base_models:
            model_name = model.replace('_RevIN', '')  # Clean model name
            first_row = True

            for method in self.online_methods:
                if first_row:
                    row = f"\\multirow{{{len(self.online_methods)}}}{{*}}{{{model_name}}} & {method}"
                    first_row = False
                else:
                    row = f" & {method}"

                for dataset in self.datasets:
                    for horizon in self.horizons:
                        key = f"{model}_{method}_{dataset}_{horizon}"
                        if key in self.results and 'mae' in self.results[key]:
                            mae_val = self.results[key]['mae']
                            row += f" & {mae_val:.3f}"
                        else:
                            row += " & -"

                row += " \\\\"
                table_lines.append(row)

        table_lines.append("\\end{tabular}")

        # Save table
        table_file = self.results_dir / "mae_table.tex"
        with open(table_file, 'w') as f:
            f.write('\n'.join(table_lines))

        print(f"MAE table saved to {table_file}")

    def run_quick_test(self):
        """Run a quick test with a subset of experiments"""
        print("Running quick test with subset of experiments...")

        # Test configuration - using available script combinations
        test_configs = [
            ('TCN_RevIN', 'ClearE', 'ETTm1', 24),  # Available
            ('TCN_RevIN', 'SOLID++', 'ETTh2', 24),  # Available
            ('PatchTST', 'ClearE', 'ECL', 24),      # Available
            ('PatchTST', 'Proceed', 'ETTm1', 24),   # Available
        ]

        for model, method, dataset, horizon in test_configs:
            print(f"\nTesting: {model} + {method} on {dataset} with horizon {horizon}")
            result = self.run_single_experiment(model, method, dataset, horizon)

            if result:
                key = f"{model}_{method}_{dataset}_{horizon}"
                self.results[key] = result
                print(f"✓ MSE: {result.get('mse', 'N/A'):.3f}, MAE: {result.get('mae', 'N/A'):.3f}")
            else:
                print("✗ Test failed")

        self._save_final_results()
        print("Quick test completed!")

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='CLEAR-E Comprehensive Online Learning Experiments')
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                       help='Experiment mode: full (all experiments) or quick (subset test)')
    parser.add_argument('--models', nargs='+', default=['TCN_RevIN', 'PatchTST', 'iTransformer'],
                       help='Base models to test')
    parser.add_argument('--methods', nargs='+', default=['Online', 'FSNet', 'OneNet', 'SOLID++', 'Proceed', 'ClearE'],
                       help='Online methods to test')
    parser.add_argument('--datasets', nargs='+', default=['ETTh2', 'ETTm1', 'Weather', 'ECL', 'Traffic'],
                       help='Datasets to test')
    parser.add_argument('--horizons', nargs='+', type=int, default=[24, 48, 96],
                       help='Prediction horizons to test')

    args = parser.parse_args()

    print("CLEAR-E Comprehensive Online Learning Experiments")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Models: {args.models}")
    print(f"Methods: {args.methods}")
    print(f"Datasets: {args.datasets}")
    print(f"Horizons: {args.horizons}")
    print("=" * 60)

    runner = ComprehensiveExperimentRunner()
    runner.base_models = args.models
    runner.online_methods = args.methods
    runner.datasets = args.datasets
    runner.horizons = args.horizons

    if args.mode == 'full':
        runner.run_all_experiments()
    else:
        runner.run_quick_test()

    print("\nExperiment suite completed successfully!")
    print(f"Results available in: {runner.results_dir}")

if __name__ == "__main__":
    main()
