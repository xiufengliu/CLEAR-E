#!/bin/bash
#BSUB -J clear_e_experiments
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 24:00
#BSUB -o logs/clear_e_experiments_%J.out
#BSUB -e logs/clear_e_experiments_%J.err
#BSUB -N

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results/clear_e_experiments
mkdir -p checkpoints

# Load necessary modules (using local conda for Python)
module load cuda/11.8

# Activate conda environment
source /work3/xiuli/anaconda3/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Print system information
echo "=== System Information ==="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Queue: $LSB_QUEUE"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "CUDA Version: $(nvcc --version 2>/dev/null || echo 'CUDA not available')"
echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
echo "=========================="

# Define experimental parameters based on available datasets
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "ECL" "gefcom2014" "southern_china")
MODELS=("PatchTST" "DLinear" "Transformer" "iTransformer")
PRED_LENS=(24 48 96)
SEQ_LEN=96
BATCH_SIZE=32
LEARNING_RATE=0.001
ONLINE_LEARNING_RATE=0.0001
EPOCHS=50
ITR=3

# CLEAR-E specific parameters
CONCEPT_DIM=64
BOTTLENECK_DIM=32
METADATA_DIM=10
METADATA_HIDDEN_DIM=32
DRIFT_MEMORY_SIZE=10
DRIFT_REG_WEIGHT=0.1
HIGH_LOAD_THRESHOLD=0.8
UNDERESTIMATE_PENALTY=2.0

echo "=== Starting CLEAR-E Experiments ==="
echo "Datasets: ${DATASETS[@]}"
echo "Models: ${MODELS[@]}"
echo "Prediction horizons: ${PRED_LENS[@]}"
echo "=================================="

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local model=$2
    local pred_len=$3
    local method=$4
    
    echo "Running experiment: Dataset=$dataset, Model=$model, Horizon=$pred_len, Method=$method"
    
    local log_file="logs/${model}_${method}_${dataset}_${pred_len}.log"
    
    if [ "$method" == "ClearE" ]; then
        python -u run.py \
            --dataset $dataset \
            --model $model \
            --seq_len $SEQ_LEN \
            --pred_len $pred_len \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --online_learning_rate $ONLINE_LEARNING_RATE \
            --train_epochs $EPOCHS \
            --itr $ITR \
            --online_method ClearE \
            --concept_dim $CONCEPT_DIM \
            --bottleneck_dim $BOTTLENECK_DIM \
            --metadata_dim $METADATA_DIM \
            --metadata_hidden_dim $METADATA_HIDDEN_DIM \
            --drift_memory_size $DRIFT_MEMORY_SIZE \
            --drift_reg_weight $DRIFT_REG_WEIGHT \
            --use_energy_loss \
            --high_load_threshold $HIGH_LOAD_THRESHOLD \
            --underestimate_penalty $UNDERESTIMATE_PENALTY \
            --border_type online \
            --pretrain \
            --save_opt \
            --only_test \
            --val_online_lr \
            --diff_online_lr \
            --tune_mode down_up \
            --features M >> $log_file 2>&1
    elif [ "$method" == "Proceed" ]; then
        python -u run.py \
            --dataset $dataset \
            --model $model \
            --seq_len $SEQ_LEN \
            --pred_len $pred_len \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --online_learning_rate $ONLINE_LEARNING_RATE \
            --train_epochs $EPOCHS \
            --itr $ITR \
            --online_method Proceed \
            --concept_dim $CONCEPT_DIM \
            --bottleneck_dim $BOTTLENECK_DIM \
            --border_type online \
            --pretrain \
            --save_opt \
            --only_test \
            --val_online_lr \
            --diff_online_lr \
            --tune_mode down_up \
            --features M >> $log_file 2>&1
    else
        # Other baseline methods
        python -u run.py \
            --dataset $dataset \
            --model $model \
            --seq_len $SEQ_LEN \
            --pred_len $pred_len \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --online_learning_rate $ONLINE_LEARNING_RATE \
            --train_epochs $EPOCHS \
            --itr $ITR \
            --online_method $method \
            --border_type online \
            --pretrain \
            --save_opt \
            --only_test \
            --features M >> $log_file 2>&1
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed: Dataset=$dataset, Model=$model, Horizon=$pred_len, Method=$method"
    else
        echo "✗ Failed: Dataset=$dataset, Model=$model, Horizon=$pred_len, Method=$method (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Counter for tracking progress
total_experiments=0
completed_experiments=0
failed_experiments=0

# Calculate total number of experiments
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for pred_len in "${PRED_LENS[@]}"; do
            total_experiments=$((total_experiments + 3))  # ClearE, Proceed, Online
        done
    done
done

echo "Total experiments to run: $total_experiments"
echo "=========================================="

# Run comprehensive experiments using Python script
echo "Running comprehensive experiments..."

# Check if mode argument is provided, default to 'quick'
MODE=${1:-quick}
echo "Experiment mode: $MODE"

# Run experiments with specified mode
python run_comprehensive_experiments.py --mode $MODE

echo "Comprehensive experiments completed!"

echo "=== Experiment Summary ==="
echo "Total experiments: $total_experiments"
echo "Completed: $completed_experiments"
echo "Failed: $failed_experiments"
echo "Success rate: $(echo "scale=2; $completed_experiments * 100 / $total_experiments" | bc)%"
echo "=========================="

echo "=== All experiments completed ==="
echo "Results saved in: results/clear_e_experiments/"
echo "Logs saved in: logs/"
echo "Job completed at: $(date)"

# Create a summary of results
echo "Creating results summary..."
python3 << 'EOF'
import os
import glob
import json
from datetime import datetime

# Collect all log files
log_files = glob.glob("logs/*.log")
summary = {
    "experiment_date": datetime.now().isoformat(),
    "total_experiments": len(log_files),
    "results": {}
}

for log_file in log_files:
    filename = os.path.basename(log_file)
    parts = filename.replace('.log', '').split('_')
    if len(parts) >= 4:
        model, method, dataset, pred_len = parts[0], parts[1], parts[2], parts[3]
        
        # Try to extract results from log file
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                # Look for result patterns (this is a simple extraction)
                if "mse:" in content and "mae:" in content:
                    lines = content.split('\n')
                    for line in lines:
                        if line.startswith('mse:') and 'mae:' in line:
                            parts = line.split(',')
                            mse = float(parts[0].split(':')[1].strip())
                            mae = float(parts[1].split(':')[1].strip())
                            
                            key = f"{dataset}_{model}_{method}_{pred_len}"
                            summary["results"][key] = {
                                "dataset": dataset,
                                "model": model,
                                "method": method,
                                "pred_len": int(pred_len),
                                "mse": mse,
                                "mae": mae
                            }
                            break
        except Exception as e:
            print(f"Error processing {log_file}: {e}")

# Save summary
with open("results/clear_e_experiments/experiment_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved with {len(summary['results'])} results")
EOF

echo "Results summary created at: results/clear_e_experiments/experiment_summary.json"
