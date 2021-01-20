import numpy as np
import os
import pandas as pd
import torch
from typing import List, Tuple

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
    preproc_dir: str, train_set_label_proportion: float, as_numpy: bool = False
) -> Tuple[torch.LongTensor, torch.LongTensor]:

    print(f'Loading training subset split for label proportion {train_set_label_proportion}')

    # choose the correct training subset
    subset_name = f'{train_set_label_proportion:{1:d}}'
    subset_dir = os.path.join(preproc_dir, f"training_set_proportion_{subset_name.replace('.', '_')}")
    train_nodes = torch.load(os.path.join(subset_dir, f'train-indices-{subset_name}.pt'))

    # validation set is the same regardless of the training subset used
    val_nodes = torch.load(os.path.join(preproc_dir, 'val-indices.pt'))

    if as_numpy:
        return train_nodes.numpy(), val_nodes.numpy()
    return train_nodes, val_nodes
