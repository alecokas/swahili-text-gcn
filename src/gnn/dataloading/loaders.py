from collections import Counter
import numpy as np
import os
from random import shuffle, sample, choices
from scipy.sparse import csr_matrix, diags, load_npz
import torch
from typing import Tuple


def load_datasets(graph_dir: str) -> Tuple[torch.sparse.FloatTensor, torch.FloatTensor, torch.LongTensor]:
    labels = torch.load(os.path.join(graph_dir, 'labels.pt'))
    input_features = torch.load(os.path.join(graph_dir, 'input_features.pt')).to_dense()
    adjacency = load_npz(os.path.join(graph_dir, 'adjacency.npz'))

    norm_adjacency = _normalise_adjacency(adjacency).tocoo()
    norm_adjacency = torch.sparse.FloatTensor(
        torch.LongTensor([norm_adjacency.row.tolist(), norm_adjacency.col.tolist()]),
        torch.FloatTensor(norm_adjacency.data),
    )

    return norm_adjacency, input_features, labels


def _normalise_adjacency(adjacency: csr_matrix) -> csr_matrix:
    """ A_tidle = D^(-0.5) A D^(-0.5) """
    # Create D^(-0.5)
    degree_inv_sqrt = np.power(np.array(adjacency.sum(1)), -0.5).flatten()
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    degree_inv_sqrt = diags(degree_inv_sqrt, format="coo")
    # Compute D^(-0.5) A D^(-0.5)
    return degree_inv_sqrt.dot(adjacency).dot(degree_inv_sqrt)


def load_train_val_nodes(
    train_dir: str, train_set_label_proportion: float
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    train_nodes = torch.load(os.path.join(train_dir, 'train-indices.pt')).tolist()
    train_nodes = torch.LongTensor(sample(train_nodes, k=int(len(train_nodes) * train_set_label_proportion)))

    val_nodes = torch.load(os.path.join(train_dir, 'val-indices.pt'))
    return train_nodes, val_nodes


def create_train_val_split(
    train_dir: str, node_labels: torch.LongTensor, train_ratio: float, val_split_type: str
) -> None:
    if val_split_type == 'balance_for_all_classes':
        _create_balanced_val_split(train_dir, node_labels, train_ratio)
    elif val_split_type == 'uniform_over_all_data':
        _create_uniformly_sampled_split(train_dir, node_labels, train_ratio)
    else:
        raise Exception(f'{val_split_type} is not a valid val_split_type')


def _create_balanced_val_split(train_dir: str, node_labels: torch.LongTensor, train_ratio: float) -> None:
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

    torch.save(torch.LongTensor(train_labels), os.path.join(train_dir, 'train-indices.pt'))
    torch.save(torch.LongTensor(val_labels), os.path.join(train_dir, 'val-indices.pt'))


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
