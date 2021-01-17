import numpy as np
import os
import pandas as pd
from random import sample
import torch
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from collections import Counter

from shared.utils import save_dict_to_json


def load_text_and_labels(df_path: str, text_column: str, label_column: str) -> Tuple[List[str], List[str]]:
    """
    Load the CSV from file, extract the column we want along with the corresponding labels.
    """
    dataset_df = pd.read_csv(df_path, sep=';')
    document_content = dataset_df[text_column].values.tolist()
    labels = dataset_df[label_column].values.tolist()
    assert len(labels) == len(document_content), 'The number of labels and docs should match'
    return document_content, labels


def save_categorical_labels(save_dir: str, labels: List[str], as_numpy: bool = False) -> None:
    label_to_cat = {label: cat for cat, label in enumerate(set(labels))}
    catagorical_labels = torch.LongTensor([label_to_cat[label] for label in labels])
    save_dict_to_json(label_to_cat, os.path.join(save_dir, 'label_map.json'))
    if as_numpy:
        np.save(os.path.join(save_dir, 'labels.npy'), catagorical_labels)
    else:
        torch.save(catagorical_labels, os.path.join(save_dir, 'labels.pt'))


def load_train_val_nodes(
    preproc_dir: str, train_set_label_proportion: float, random_state: int, as_numpy: bool = False
) -> Tuple[torch.LongTensor, torch.LongTensor]:

    train_nodes = torch.load(os.path.join(preproc_dir, 'train-indices.pt'))
    train_labels = torch.load(os.path.join(preproc_dir, 'train-labels.pt'))
    val_nodes = torch.load(os.path.join(preproc_dir, 'val-indices.pt'))

    if train_set_label_proportion == 0:
        if as_numpy:
            return (train_nodes_subset.numpy(), val_nodes.numpy())
        return train_nodes_subset, val_nodes

    train_nodes_subset, _, train_label_subset, _ = train_test_split(
        train_nodes,
        train_labels,
        stratify=train_labels,
        test_size=1 - train_set_label_proportion,
        random_state=random_state,
    )

    print(f'Training subset split: {dict(Counter(train_label_subset.numpy()))}')

    if as_numpy:
        return (train_nodes_subset.numpy(), val_nodes.numpy())
    return train_nodes_subset, val_nodes
