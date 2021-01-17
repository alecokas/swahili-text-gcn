from collections import Counter
import os
from random import shuffle, sample
import shutil
import torch
from typing import List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from shared.utils import save_dict_to_json


def create_train_val_split(
    results_dir: str, node_labels: torch.LongTensor, train_ratio: float, val_split_type: str
) -> None:
    if val_split_type == 'balance_for_all_classes':
        _create_balanced_val_split(results_dir, node_labels, train_ratio)
    elif val_split_type == 'uniform_over_all_data':
        _create_uniformly_sampled_split(results_dir, node_labels, train_ratio)
    else:
        raise Exception(f'{val_split_type} is not a valid val_split_type')


def new_create_train_val_split(
    results_dir: str, df: pd.DataFrame, train_ratio: float, random_state: int
) -> None:
    train_nodes, val_nodes, train_labels, val_labels = train_test_split(
        df.document_idx, df.label_idx, stratify=df.label_idx, test_size=1 - train_ratio, random_state=random_state
    )
    names = ['train-indices', 'val-indices', 'train-labels', 'val-labels']
    for name, data in zip(names, [train_nodes, val_nodes, train_labels, val_labels]):
        torch.save(torch.LongTensor(data.values), os.path.join(results_dir, f'{name}.pt'))
    _subset_distribution(df.label_idx, train_nodes.values, val_nodes.values, results_dir)


def copy_truth_data_split(data_split_dir: str, results_dir: str, cat_labels: torch.LongTensor) -> None:
    # First check that the number of labels add up
    num_train_nodes = len(torch.load(os.path.join(data_split_dir, 'train-indices.pt')).tolist())
    num_val_nodes = len(torch.load(os.path.join(data_split_dir, 'val-indices.pt')).tolist())
    num_labels = len(cat_labels.tolist())
    assert num_train_nodes + num_val_nodes == num_labels, f'Expected {num_train_nodes + num_val_nodes} == {num_labels}'
    # Copy to new directory
    print(f'Copying train and val indices from {data_split_dir}')
    shutil.copy(os.path.join(data_split_dir, 'train-indices.pt'), os.path.join(results_dir, 'train-indices.pt'))
    shutil.copy(os.path.join(data_split_dir, 'val-indices.pt'), os.path.join(results_dir, 'val-indices.pt'))


def _create_balanced_val_split(results_dir: str, node_labels: torch.LongTensor, train_ratio: float) -> None:
    """ Set up a validation set which has a balanced split over all classes, even if the data is not evenly split """
    total_num_nodes = len(node_labels)
    unique_labels = node_labels.unique()
    num_val_nodes_per_label = int(total_num_nodes * (1 - train_ratio) / len(unique_labels))

    val_indices = []
    for label in unique_labels:
        indices = [idx for idx in list(range(total_num_nodes)) if node_labels[idx] == label]
        shuffle(indices)
        val_indices.extend(indices[:num_val_nodes_per_label])
    shuffle(val_indices)

    assert len(val_indices) == (
        num_val_nodes_per_label * len(unique_labels)
    ), f'Expected {len(val_indices)} == {num_val_nodes_per_label * len(unique_labels)}'

    val_set = set(val_indices)
    train_indices = [index for index in range(total_num_nodes) if index not in val_set]
    shuffle(train_indices)

    assert (
        len(train_indices) + len(val_indices)
    ) == total_num_nodes, f'Not all indices are included in the split: \
        Expected {len(train_indices) + len(val_indices)} == {total_num_nodes}'

    _subset_distribution(node_labels, train_indices, val_indices, results_dir)
    _save_indices(train_indices, val_indices, results_dir)


def _create_uniformly_sampled_split(train_dir: str, node_labels: torch.LongTensor, train_ratio: float) -> None:
    """ Uniformly sample to generate a validation set with approx. the same distribution as the full dataset """
    total_num_nodes = len(node_labels)
    num_val_nodes = int(total_num_nodes * (1 - train_ratio))
    val_indices = sample(list(range(total_num_nodes)), k=num_val_nodes)

    val_set = set(val_indices)
    train_indices = [index for index in range(total_num_nodes) if index not in val_set]

    assert (
        len(train_indices) + len(val_indices)
    ) == total_num_nodes, f'Not all indices are included in the split: \
        Expected {len(train_indices) + len(val_indices)} == {total_num_nodes}'

    _subset_distribution(node_labels, train_indices, val_indices, train_dir)
    _save_indices(train_indices, val_indices, train_dir)


def _subset_distribution(
    node_labels: torch.LongTensor, train_indices: List[int], val_indices: List[int], results_dir: str
) -> None:
    """ Check the subset class distribution and save indices """
    train_labels = node_labels[train_indices].tolist()
    val_labels = node_labels[val_indices].tolist()
    print(f'Training set split: {dict(Counter(train_labels))}')
    print(f'Validation set split: {Counter(val_labels)}')
    save_dict_to_json(dict(Counter(train_labels)), os.path.join(results_dir, 'train-label-dist.json'))
    save_dict_to_json(dict(Counter(val_labels)), os.path.join(results_dir, 'val-label-dist.json'))


def _save_indices(train_labels: torch.LongTensor, val_labels: torch.LongTensor, results_dir: str) -> None:
    torch.save(torch.LongTensor(train_labels), os.path.join(results_dir, 'train-indices.pt'))
    torch.save(torch.LongTensor(val_labels), os.path.join(results_dir, 'val-indices.pt'))
