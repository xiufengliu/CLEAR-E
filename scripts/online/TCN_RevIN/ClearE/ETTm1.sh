#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=60
data=ETTm1
model_name=TCN
online_method=ClearE
tune_mode=down_up

for mid in 150
do
for btl in 48
do
for pred_len in 24
do
for learning_rate in 0.001
do
for online_learning_rate in 0.00001
do
  suffix='_lr'$learning_rate'_onlinelr'$online_learning_rate
  filename=logs/online/$model_name'_RevIN_'$online_method'_mid'$mid'_share_fulltune_'$tune_mode'_'$data'_'$pred_len'_btl'$btl$suffix.log
  python -u run.py \
    --dataset $data --border_type 'online' --batch_size 16 \
    --model $model_name --normalization RevIN \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --tune_mode $tune_mode \
    --val_online_lr --diff_online_lr \
    --itr 3 --skip $filename --pretrain \
    --save_opt --online_method $online_method \
    --bottleneck_dim $btl --concept_dim $mid \
    --learning_rate $learning_rate \
    --online_learning_rate $online_learning_rate \
    --metadata_dim 10 --metadata_hidden_dim 32 \
    --drift_memory_size 10 --drift_reg_weight 0.1 \
    --use_energy_loss --high_load_threshold 0.8 \
    --underestimate_penalty 2.0 \
    >> $filename 2>&1
done
done
done
done
done
