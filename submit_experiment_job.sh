#!/bin/bash
#BSUB -J clear_e_experiments
#BSUB -o logs/clear_e_experiments_%J.out
#BSUB -e logs/clear_e_experiments_%J.err
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

# First run a test to verify fixes
echo "=== Running Test Experiments ==="
python test_experiment_fixes.py

# Check if test was successful
if [ $? -eq 0 ]; then
    echo "=== Starting CLEAR-E Experiments ==="
    # Print experiment configuration
    python -c "import run_comprehensive_experiments as rce; print('Datasets:', rce.main.__globals__['datasets']); print('Models:', rce.main.__globals__['models']); print('Prediction horizons:', rce.main.__globals__['pred_lens']); print('Methods:', rce.main.__globals__['methods'])"
    
    # Calculate total experiments
    TOTAL=$(python -c "import run_comprehensive_experiments as rce; datasets=rce.main.__globals__['datasets']; models=rce.main.__globals__['models']; pred_lens=rce.main.__globals__['pred_lens']; methods=rce.main.__globals__['methods']; print(len(datasets) * len(models) * len(pred_lens) * len(methods))")
    echo "Total experiments to run: $TOTAL"
    echo "=========================================="
    
    # Run the comprehensive experiments
    echo "Running comprehensive experiments..."
    python run_comprehensive_experiments.py
else
    echo "Test experiments failed. Please fix the issues before running the full experiment suite."
    exit 1
fi
