#!/bin/bash

python bert_w_lightning.py \
    --model_size "small" \
    --device "mps" \
    --project_name "my_project" \
    --output_dir "output" \
    --train_data_path "data/train_data.csv" \
    --valid_data_path "data/valid_data.csv" \
    --test_data_path "data/test_data.csv" \
    --target_field "my_target" \
    --num_targets 10 \
    --text_field "my_text" \
    --batch_size 64 \
    --learning_rate "1e-5" \
    --max_epochs 100 \
    --weight_decay 0.005 \
    --dropout 0.8 \

