from collections import Counter
import os
import torch
from typing import List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from shared.utils import save_dict_to_json


def create_train_val_split(
    results_dir: str,
    df: pd.DataFrame,
    train_ratio: float,
    random_state: int,
    train_set_label_proportions: Optional[List[float]] = None,
) -> None:

    if train_set_label_proportions is None:
        train_set_label_proportions = [0.01, 0.05, 0.1, 0.2]

    train_nodes, val_nodes, train_labels, val_labels = train_test_split(
        df.document_idx.values,
        df.label_idx.values,
        stratify=df.label_idx.values,
        test_size=1 - train_ratio,
        random_state=random_state,
    )
    val_nodes, test_nodes, val_labels, test_labels = train_test_split(
        val_nodes,
        val_labels,
        stratify=val_labels,
        test_size=0.5,
        random_state=random_state,
    )

    names = ['train-indices', 'val-indices', 'test-indices', 'train-labels', 'val-labels', 'test-labels']

    assert (len(train_nodes) + len(val_nodes) + len(test_nodes)) == len(
        df
    ), f'Not all indices are included in the split: \
        Expected {len(train_nodes) + len(val_nodes) + len(test_nodes)} == {len(df)}'

    for name, data in zip(names, [train_nodes, val_nodes, test_nodes, train_labels, val_labels, test_labels]):
        torch.save(torch.LongTensor(data), os.path.join(results_dir, f'{name}.pt'))
    _subset_distribution(df.label_idx.values, train_nodes, results_dir, 'train')
    _subset_distribution(df.label_idx.values, val_nodes, results_dir, 'val')
    _subset_distribution(df.label_idx.values, test_nodes, results_dir, 'test')

    for train_set_label_proportion in train_set_label_proportions:
        subset_name = f'{train_set_label_proportion:{1:d}}'
        subset_dir = os.path.join(results_dir, f"training_set_proportion_{subset_name.replace('.', '_')}")
        os.makedirs(subset_dir, exist_ok=True)

        train_nodes_subset, _, train_label_subset, _ = train_test_split(
            train_nodes,
            train_labels,
            stratify=train_labels,
            test_size=1 - train_set_label_proportion,
            random_state=random_state,
        )

        torch.save(torch.LongTensor(train_nodes_subset), os.path.join(subset_dir, f'train-indices-{subset_name}.pt'))
        torch.save(torch.LongTensor(train_label_subset), os.path.join(subset_dir, f'train-labels-{subset_name}.pt'))


def _subset_distribution(node_labels: torch.LongTensor, node_indices: List[int], results_dir: str, name: str) -> None:
    """ Check the subset class distribution and save indices """
    labels = node_labels[node_indices].tolist()
    print(f'{name} set split: {dict(Counter(labels))}')
    save_dict_to_json(dict(Counter(labels)), os.path.join(results_dir, f'{name}-label-dist.json'))


def _save_indices(train_labels: torch.LongTensor, val_labels: torch.LongTensor, results_dir: str) -> None:
    torch.save(torch.LongTensor(train_labels), os.path.join(results_dir, 'train-indices.pt'))
    torch.save(torch.LongTensor(val_labels), os.path.join(results_dir, 'val-indices.pt'))
