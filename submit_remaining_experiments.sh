#!/bin/bash
#BSUB -J remaining_experiments
#BSUB -o logs/remaining_experiments_%J.out
#BSUB -e logs/remaining_experiments_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00

# Print system information
echo "=== System Information ==="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Queue: $LSB_QUEUE"
echo "Host: $HOSTNAME"
echo "Date: $(date)"
echo "Working Directory: $PWD"
echo "Python Version: $(python --version)"
echo "CUDA Version: $(nvcc --version)"
echo "GPU Information:"
nvidia-smi
echo "=========================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Create results directory if it doesn't exist
mkdir -p results/remaining_experiments

echo "=== Starting Remaining Datasets Experiments ==="
echo "Datasets: ECL, gefcom2014, southern_china"
echo "Models: All 18 available models (PatchTST, iTransformer, Autoformer, Informer, Transformer, Crossformer, DLinear, Linear, NLinear, RLinear, TCN, TCN_RevIN, MTGNN, GPT4TS, FSNet, OneNet, LIFT, LightMTS)"
echo "Prediction horizons: 24, 48, 96"
echo "Methods: Offline, Online, FSNet, OneNet, Proceed, ClearE"
echo "=========================================="

# Calculate total experiments
TOTAL=$(python -c "datasets=['ECL', 'gefcom2014', 'southern_china']; models=['PatchTST', 'iTransformer', 'Autoformer', 'Informer', 'Transformer', 'Crossformer', 'DLinear', 'Linear', 'NLinear', 'RLinear', 'TCN', 'TCN_RevIN', 'MTGNN', 'GPT4TS', 'FSNet', 'OneNet', 'LIFT', 'LightMTS']; pred_lens=[24, 48, 96]; methods=['Offline', 'Online', 'FSNet', 'OneNet', 'Proceed', 'ClearE']; print(len(datasets) * len(models) * len(pred_lens) * len(methods))")
echo "Total experiments to run: $TOTAL"
echo "=========================================="

# Run the remaining experiments
echo "Running remaining experiments..."
python run_remaining_experiments.py

echo "=== Experiments completed ==="
echo "Job completed at: $(date)"

# Create a summary of results
echo "Creating results summary..."
python -c "
import json
import glob
from pathlib import Path

results_dir = Path('results/remaining_experiments')
final_files = list(results_dir.glob('final_results_*.json'))

if final_files:
    latest_file = max(final_files, key=lambda x: x.stat().st_mtime)
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    print(f'Results summary from: {latest_file}')
    print(f'Total experiments: {data[\"total_experiments\"]}')
    print(f'Successful: {data[\"successful_experiments\"]}')
    print(f'Failed: {data[\"failed_experiments\"]}')
    print(f'Success rate: {data[\"success_rate\"]:.1f}%')
    print(f'Duration: {data[\"total_duration\"]/3600:.2f} hours')
    
    # Save summary
    summary = {
        'experiment_date': '$(date -Iseconds)',
        'total_experiments': data['total_experiments'],
        'successful_experiments': data['successful_experiments'],
        'failed_experiments': data['failed_experiments'],
        'success_rate': data['success_rate'],
        'duration_hours': data['total_duration']/3600,
        'datasets': ['ECL', 'gefcom2014', 'southern_china'],
        'results_file': str(latest_file)
    }
    
    with open('results/remaining_experiments/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('Summary saved to: results/remaining_experiments/experiment_summary.json')
else:
    print('No final results found')
"

echo "Results summary created"
