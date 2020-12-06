from collections import Counter
import os
from random import shuffle, sample
import torch


def create_train_val_split(
    results_dir: str, node_labels: torch.LongTensor, train_ratio: float, val_split_type: str
) -> None:
    if val_split_type == 'balance_for_all_classes':
        _create_balanced_val_split(results_dir, node_labels, train_ratio)
    elif val_split_type == 'uniform_over_all_data':
        _create_uniformly_sampled_split(results_dir, node_labels, train_ratio)
    else:
        raise Exception(f'{val_split_type} is not a valid val_split_type')


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

    # Check the subset class distribution
    train_labels = node_labels[train_indices].tolist()
    val_labels = node_labels[val_indices].tolist()
    print(f'Training set split: {Counter(train_labels)}')
    print(f'Validation set split: {Counter(val_labels)}')

    torch.save(torch.LongTensor(train_labels), os.path.join(results_dir, 'train-indices.pt'))
    torch.save(torch.LongTensor(val_labels), os.path.join(results_dir, 'val-indices.pt'))


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

    # Check the subset class distribution
    train_labels = node_labels[train_indices].tolist()
    val_labels = node_labels[val_indices].tolist()
    print(f'Training set split: {Counter(train_labels)}')
    print(f'Validation set split: {Counter(val_labels)}')

    torch.save(torch.LongTensor(train_indices), os.path.join(train_dir, 'train-indices.pt'))
    torch.save(torch.LongTensor(val_indices), os.path.join(train_dir, 'val-indices.pt'))
