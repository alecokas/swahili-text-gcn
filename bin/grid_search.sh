#!/usr/bin/env bash

# Execute a grid search over LR and dropout proportion


name=$1
input_data_dir=$2
epochs=$3
stemmer_path=$4
lr=$5
dropout_ratio=$6
label_proportion=$7

for dropout_ratio in 0.3, 0.5
do 
    for lr in 0.002
    do 
        python src/train_gnn.py "$name"_grid_search --train-dir train_lr_"$lr"_dropout_"$dropout_ratio" --input-data-dir "$input_data_dir" \
        --input-features one-hot --stemmer-path "$stemmer_path" --epochs $epochs --lr $lr --dropout-ratio $dropout_ratio \
        --train-set-label-proportion $label_proportion
    done
done
