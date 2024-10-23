#!/bin/bash

current_time=$(date -u -d "+9 hours" "+%Y%m%d_%H%M%S")

# Set up directories
train_dir="models/train_${current_time}"
eval_dir="output/eval_${current_time}"
predict_dir="output/test_${current_time}"
predict_dataset_name="data/test_dataset"

#cd /data/ephemeral/home/level2-mrc-nlp-15/src

# Perform training
python src/main.py --output_dir $train_dir --do_train --max_seq_length 384 --per_device_train_batch_size 16 --num_train_epochs 3 --learning_rate "1e-5"
