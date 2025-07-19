#!/bin/bash
#BSUB -J clear_e_quick_test
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 2
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 4:00
#BSUB -o logs/clear_e_quick_test_%J.out
#BSUB -e logs/clear_e_quick_test_%J.err
#BSUB -N

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results/quick_test
mkdir -p checkpoints

# Load necessary modules
module load python3/3.8.5
module load cuda/11.2
module load cudnn/v8.1.0.77-prod-cuda-11.2

# Activate conda environment
source /work3/xiuli/anaconda3/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== Quick Test - System Information ==="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not available')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Cannot check CUDA')"
echo "========================================"

# Quick test parameters - reduced for faster execution
DATASET="ETTh1"
MODEL="DLinear"
PRED_LEN=24
SEQ_LEN=96
BATCH_SIZE=16
LEARNING_RATE=0.001
ONLINE_LEARNING_RATE=0.0001
EPOCHS=5
ITR=1

# CLEAR-E parameters
CONCEPT_DIM=32
BOTTLENECK_DIM=16
METADATA_DIM=10
METADATA_HIDDEN_DIM=16
DRIFT_MEMORY_SIZE=5
DRIFT_REG_WEIGHT=0.1

echo "=== Quick Test Configuration ==="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Prediction Length: $PRED_LEN"
echo "Epochs: $EPOCHS"
echo "Iterations: $ITR"
echo "==============================="

# Function to run a single test
run_test() {
    local method=$1
    local log_file="logs/quick_test_${method}_${DATASET}_${MODEL}_${PRED_LEN}.log"
    
    echo "Testing method: $method"
    
    if [ "$method" == "ClearE" ]; then
        python -u run.py \
            --dataset $DATASET \
            --model $MODEL \
            --seq_len $SEQ_LEN \
            --pred_len $PRED_LEN \
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
            --high_load_threshold 0.8 \
            --underestimate_penalty 2.0 \
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
            --dataset $DATASET \
            --model $MODEL \
            --seq_len $SEQ_LEN \
            --pred_len $PRED_LEN \
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
        # Baseline method
        python -u run.py \
            --dataset $DATASET \
            --model $MODEL \
            --seq_len $SEQ_LEN \
            --pred_len $PRED_LEN \
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
        echo "✓ $method test completed successfully"
        # Extract results from log
        if grep -q "mse:" $log_file; then
            echo "Results found in $log_file:"
            grep "mse:" $log_file | tail -1
        fi
    else
        echo "✗ $method test failed (exit code: $exit_code)"
        echo "Check log file: $log_file"
    fi
    
    return $exit_code
}

# Run quick tests
echo "=== Starting Quick Tests ==="

# Test CLEAR-E
run_test "ClearE"

# Test PROCEED baseline
run_test "Proceed"

# Test Online baseline
run_test "Online"

echo "=== Quick Test Summary ==="
echo "Check individual log files in logs/ directory for detailed results"
echo "Quick test completed at: $(date)"

# Simple results extraction
echo "=== Extracting Results ==="
for log_file in logs/quick_test_*.log; do
    if [ -f "$log_file" ]; then
        method=$(basename "$log_file" | cut -d'_' -f3)
        echo "Method: $method"
        if grep -q "mse:" "$log_file"; then
            grep "mse:" "$log_file" | tail -1
        else
            echo "  No results found"
        fi
        echo "---"
    fi
done

echo "Quick test job completed!"
