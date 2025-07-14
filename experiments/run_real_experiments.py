#!/usr/bin/env python3
"""
Run Real CLEAR-E Experiments
This script runs the actual experimental evaluation to replace synthetic results
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from experiments.real_experimental_framework import RealExperimentalFramework

def create_experiment_config():
    """Create experimental configuration"""
    config = {
        "datasets": ["ETTh1", "ETTh2", "southern_china"],
        "lookback": 96,
        "horizon": 24,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        "patience": 15,
        "n_runs": 5,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "baseline_models": ["DLinear", "PatchTST", "Transformer"],
        "clear_e_config": {
            "concept_dim": 64,
            "hidden_dim": 128,
            "bottleneck_dim": 32,
            "memory_size": 10,
            "momentum": 0.9,
            "penalty_weight": 1.4,
            "drift_reg_weight": 0.1,
            "adaptive_memory": True,
            "drift_threshold": 0.5
        }
    }
    return config

def run_quick_test():
    """Run a quick test with reduced parameters"""
    print("Running quick test...")
    
    config = {
        "datasets": ["ETTh1"],
        "lookback": 48,
        "horizon": 12,
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 5,
        "patience": 3,
        "n_runs": 1,
        "train_ratio": 0.7,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "baseline_models": ["DLinear"],
        "clear_e_config": {
            "concept_dim": 32,
            "hidden_dim": 64,
            "bottleneck_dim": 16,
            "memory_size": 5,
            "momentum": 0.9,
            "penalty_weight": 1.4,
            "drift_reg_weight": 0.1,
            "adaptive_memory": True,
            "drift_threshold": 0.5
        }
    }
    
    # Save config
    with open("quick_test_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run experiment
    framework = RealExperimentalFramework("quick_test_config.json")
    results = framework.run_experiments()
    framework.save_results("quick_test_results")
    
    print("Quick test completed!")
    return results

def run_full_experiments():
    """Run full experimental evaluation"""
    print("Running full experimental evaluation...")
    
    config = create_experiment_config()
    
    # Save config
    with open("full_experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run experiment
    framework = RealExperimentalFramework("full_experiment_config.json")
    results = framework.run_experiments()
    framework.save_results("full_experiment_results")
    
    print("Full experimental evaluation completed!")
    return results

def update_paper_results(results_dir: str):
    """Update paper with real experimental results"""
    print(f"Updating paper with results from {results_dir}...")
    
    # Load analysis results
    analysis_file = os.path.join(results_dir, "analysis.json")
    if not os.path.exists(analysis_file):
        print(f"Analysis file not found: {analysis_file}")
        return
    
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    # Create LaTeX table content for paper
    latex_content = generate_latex_table(analysis)
    
    # Save LaTeX content
    with open(os.path.join(results_dir, "paper_table.tex"), 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to {results_dir}/paper_table.tex")
    print("You can now copy this content to your paper!")

def generate_latex_table(analysis):
    """Generate LaTeX table for paper"""
    latex = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison on Real Energy Datasets}
\\label{tab:performance_comparison}
\\begin{tabular}{llcccc}
\\toprule
Dataset & Model & RMSE & MAE & MAPE (\\%) & Peak Load Error (\\%) \\\\
\\midrule
"""
    
    for dataset_name, dataset_analysis in analysis.items():
        first_row = True
        for model_name, stats in dataset_analysis.items():
            if first_row:
                dataset_display = dataset_name.replace('_', '\\_')
                first_row = False
            else:
                dataset_display = ""
            
            rmse = stats.get('rmse', {})
            mae = stats.get('mae', {})
            mape = stats.get('mape', {})
            peak_error = stats.get('peak_load_error', {})
            
            latex += f"{dataset_display} & {model_name} & "
            latex += f"{rmse.get('mean', 0):.4f} & "
            latex += f"{mae.get('mean', 0):.4f} & "
            latex += f"{mape.get('mean', 0):.2f} & "
            latex += f"{peak_error.get('mean', 0):.2f} \\\\\n"
        
        if dataset_name != list(analysis.keys())[-1]:
            latex += "\\midrule\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    return latex

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run CLEAR-E Real Experiments")
    parser.add_argument("--mode", choices=["quick", "full", "update"], default="quick",
                       help="Experiment mode: quick test, full evaluation, or update paper")
    parser.add_argument("--results-dir", default="full_experiment_results",
                       help="Results directory for update mode")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        run_quick_test()
    elif args.mode == "full":
        run_full_experiments()
    elif args.mode == "update":
        update_paper_results(args.results_dir)

if __name__ == "__main__":
    main()
