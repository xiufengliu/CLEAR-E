#!/usr/bin/env python3
"""
Generate LaTeX tables for the paper from experimental results
"""

import json
import numpy as np
from pathlib import Path

def load_results():
    """Load experimental results from JSON files"""
    results_dir = Path("results/comprehensive_experiments")
    
    # Find the most recent results file
    result_files = list(results_dir.glob("results_*.json"))
    if not result_files:
        print("No results files found!")
        return None
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return data['results']

def create_mse_table(results):
    """Create MSE results table in LaTeX format"""
    
    # Group results by dataset and method
    data = {}
    for result in results:
        if not result['success']:
            continue
            
        dataset = result['dataset']
        method = result['method']
        pred_len = result['pred_len']
        mse = result['mse']
        
        if dataset not in data:
            data[dataset] = {}
        if method not in data[dataset]:
            data[dataset][method] = {}
        if pred_len not in data[dataset][method]:
            data[dataset][method][pred_len] = []
            
        data[dataset][method][pred_len].append(mse)
    
    # Calculate means and stds
    summary = {}
    for dataset in data:
        summary[dataset] = {}
        for method in data[dataset]:
            summary[dataset][method] = {}
            for pred_len in data[dataset][method]:
                values = data[dataset][method][pred_len]
                if values:
                    summary[dataset][method][pred_len] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
    
    # Generate LaTeX table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{MSE Results for CLEAR-E and Baseline Methods}")
    latex.append("\\label{tab:mse_results}")
    latex.append("\\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{Method} & \\multicolumn{3}{c|}{ETTh1} & \\multicolumn{3}{c|}{ETTh2} & \\multicolumn{3}{c|}{ETTm1} & \\multicolumn{3}{c|}{ETTm2} & \\multicolumn{3}{c}{ECL} \\\\")
    latex.append("& 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 \\\\")
    latex.append("\\hline")
    
    methods = ['Offline', 'Online', 'Proceed', 'ClearE']
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL']
    pred_lens = [24, 48, 96]
    
    for method in methods:
        row = [method]
        for dataset in datasets:
            for pred_len in pred_lens:
                if (dataset in summary and 
                    method in summary[dataset] and 
                    pred_len in summary[dataset][method]):
                    
                    stats = summary[dataset][method][pred_len]
                    if stats['count'] > 1:
                        cell = f"{stats['mean']:.3f}±{stats['std']:.3f}"
                    else:
                        cell = f"{stats['mean']:.3f}"
                else:
                    cell = "-"
                row.append(cell)
        
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def create_mae_table(results):
    """Create MAE results table in LaTeX format"""
    
    # Group results by dataset and method
    data = {}
    for result in results:
        if not result['success']:
            continue
            
        dataset = result['dataset']
        method = result['method']
        pred_len = result['pred_len']
        mae = result['mae']
        
        if dataset not in data:
            data[dataset] = {}
        if method not in data[dataset]:
            data[dataset][method] = {}
        if pred_len not in data[dataset][method]:
            data[dataset][method][pred_len] = []
            
        data[dataset][method][pred_len].append(mae)
    
    # Calculate means and stds
    summary = {}
    for dataset in data:
        summary[dataset] = {}
        for method in data[dataset]:
            summary[dataset][method] = {}
            for pred_len in data[dataset][method]:
                values = data[dataset][method][pred_len]
                if values:
                    summary[dataset][method][pred_len] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
    
    # Generate LaTeX table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{MAE Results for CLEAR-E and Baseline Methods}")
    latex.append("\\label{tab:mae_results}")
    latex.append("\\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{Method} & \\multicolumn{3}{c|}{ETTh1} & \\multicolumn{3}{c|}{ETTh2} & \\multicolumn{3}{c|}{ETTm1} & \\multicolumn{3}{c|}{ETTm2} & \\multicolumn{3}{c}{ECL} \\\\")
    latex.append("& 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 \\\\")
    latex.append("\\hline")
    
    methods = ['Offline', 'Online', 'Proceed', 'ClearE']
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL']
    pred_lens = [24, 48, 96]
    
    for method in methods:
        row = [method]
        for dataset in datasets:
            for pred_len in pred_lens:
                if (dataset in summary and 
                    method in summary[dataset] and 
                    pred_len in summary[dataset][method]):
                    
                    stats = summary[dataset][method][pred_len]
                    if stats['count'] > 1:
                        cell = f"{stats['mean']:.3f}±{stats['std']:.3f}"
                    else:
                        cell = f"{stats['mean']:.3f}"
                else:
                    cell = "-"
                row.append(cell)
        
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def create_improvement_table(results):
    """Create improvement table showing CLEAR-E vs baselines"""
    
    # Group results by dataset and method
    data = {}
    for result in results:
        if not result['success']:
            continue
            
        dataset = result['dataset']
        method = result['method']
        pred_len = result['pred_len']
        mse = result['mse']
        mae = result['mae']
        
        if dataset not in data:
            data[dataset] = {}
        if method not in data[dataset]:
            data[dataset][method] = {}
        if pred_len not in data[dataset][method]:
            data[dataset][method][pred_len] = {'mse': [], 'mae': []}
            
        data[dataset][method][pred_len]['mse'].append(mse)
        data[dataset][method][pred_len]['mae'].append(mae)
    
    # Calculate improvements
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{CLEAR-E Improvement over Baseline Methods (\\% reduction in MSE)}")
    latex.append("\\label{tab:improvements}")
    latex.append("\\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc}")
    latex.append("\\hline")
    latex.append("\\multirow{2}{*}{vs. Method} & \\multicolumn{3}{c|}{ETTh1} & \\multicolumn{3}{c|}{ETTh2} & \\multicolumn{3}{c|}{ETTm1} & \\multicolumn{3}{c|}{ETTm2} & \\multicolumn{3}{c}{ECL} \\\\")
    latex.append("& 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 & 24 & 48 & 96 \\\\")
    latex.append("\\hline")
    
    baselines = ['Offline', 'Online', 'Proceed']
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL']
    pred_lens = [24, 48, 96]
    
    for baseline in baselines:
        row = [f"vs. {baseline}"]
        for dataset in datasets:
            for pred_len in pred_lens:
                if (dataset in data and 
                    'ClearE' in data[dataset] and 
                    baseline in data[dataset] and
                    pred_len in data[dataset]['ClearE'] and
                    pred_len in data[dataset][baseline]):
                    
                    clear_e_mse = np.mean(data[dataset]['ClearE'][pred_len]['mse'])
                    baseline_mse = np.mean(data[dataset][baseline][pred_len]['mse'])
                    
                    if baseline_mse > 0:
                        improvement = ((baseline_mse - clear_e_mse) / baseline_mse) * 100
                        cell = f"{improvement:.1f}\\%"
                    else:
                        cell = "-"
                else:
                    cell = "-"
                row.append(cell)
        
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def main():
    print("Generating LaTeX tables for paper...")
    
    # Load results
    results = load_results()
    if not results:
        return
    
    print(f"Loaded {len(results)} experimental results")
    successful_results = [r for r in results if r['success']]
    print(f"Successful results: {len(successful_results)}")
    
    # Create output directory
    output_dir = Path("paper/generated_tables")
    output_dir.mkdir(exist_ok=True)
    
    # Generate tables
    print("Generating MSE table...")
    mse_table = create_mse_table(successful_results)
    with open(output_dir / "mse_table.tex", 'w') as f:
        f.write(mse_table)
    
    print("Generating MAE table...")
    mae_table = create_mae_table(successful_results)
    with open(output_dir / "mae_table.tex", 'w') as f:
        f.write(mae_table)
    
    print("Generating improvement table...")
    improvement_table = create_improvement_table(successful_results)
    with open(output_dir / "improvement_table.tex", 'w') as f:
        f.write(improvement_table)
    
    print(f"Tables saved to: {output_dir}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    datasets = set(r['dataset'] for r in successful_results)
    methods = set(r['method'] for r in successful_results)
    print(f"Datasets: {sorted(datasets)}")
    print(f"Methods: {sorted(methods)}")
    
    for dataset in sorted(datasets):
        for method in sorted(methods):
            dataset_method_results = [r for r in successful_results 
                                    if r['dataset'] == dataset and r['method'] == method]
            if dataset_method_results:
                mse_values = [r['mse'] for r in dataset_method_results]
                mae_values = [r['mae'] for r in dataset_method_results]
                print(f"{dataset}-{method}: MSE={np.mean(mse_values):.4f}±{np.std(mse_values):.4f}, "
                      f"MAE={np.mean(mae_values):.4f}±{np.std(mae_values):.4f} (n={len(mse_values)})")

if __name__ == "__main__":
    main()
