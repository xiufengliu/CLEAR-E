#!/bin/bash
#BSUB -J comprehensive_journal_experiments
#BSUB -o logs/comprehensive_journal_%J.out
#BSUB -e logs/comprehensive_journal_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 23:00

echo "=== COMPREHENSIVE JOURNAL EXPERIMENTS ==="
echo "Job ID: $LSB_JOBID"
echo "Target: 1,134 experiments with >90% success rate"
echo "Expected duration: 24-48 hours"
echo "=========================================="

# Create directories
mkdir -p logs
mkdir -p results/comprehensive_journal_experiments

# Run comprehensive experiments
python run_comprehensive_journal_experiments.py

echo "=== Experiments completed ==="
echo "Job completed at: $(date)"
