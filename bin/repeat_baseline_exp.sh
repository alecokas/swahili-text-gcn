#!/usr/bin/env bash

# Repeat baseline experiment with different random seeds

name=$1
model=$2
input_data_dir=$3
stemmer_path=$4
label_proportion=$5
doc2vec_dm=$6


for seed in 12321 19291 48911 1403 1912
do
    python src/train_baseline.py "$name"_seed_"$seed" --model "$model" --input-data-dir "$input_data_dir" \
    --stemmer-path "$stemmer_path" --train-set-label-proportion $label_proportion --doc2vec-dm $doc2vec_dm \
    --seed $seed
done