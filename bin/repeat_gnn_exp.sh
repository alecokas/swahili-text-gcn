#!/usr/bin/env bash

# Run the one-hot GCN experiment a number of times with different seeds

name=$1
input_data_dir=$2
stemmer_path=$3
epochs=$4
lr=$5
dropout_ratio=$6
label_proportion=$7

for seed in 12321 19291 48911 1403 1912
do 
    python src/train_gnn.py "$name"_repeat_gnn_exp --input-data-dir "$input_data_dir" --input-features one-hot \
    --stemmer-path "$stemmer_path" --train-dir train_"$seed" --epochs $epochs --lr $lr --dropout-ratio $dropout_ratio \
    --train-set-label-proportion $label_proportion --seed $seed
done