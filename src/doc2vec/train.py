from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import pandas as pd
import torch
from typing import List, Tuple


def read_and_format_docs(df_path: str, text_column: str) -> List[TaggedDocument]:
    dataset_df = pd.read_csv(df_path, sep=';')
    docs_list = dataset_df[text_column].values.tolist()
    return [TaggedDocument(doc, [i]) for i, doc in enumerate(docs_list)]


def separate_into_subsets(
    docs: List[TaggedDocument], input_dir: str
) -> Tuple[List[TaggedDocument], List[TaggedDocument]]:
    """ Separate the docs into the predefined train and val lists """
    train_indices = torch.load(os.path.join(input_dir, 'train-indices.pt')).tolist()
    val_indices = torch.load(os.path.join(input_dir, 'val-indices.pt')).tolist()

    train_docs = [docs[idx] for idx in train_indices]
    val_docs = [docs[idx] for idx in val_indices]

    return train_docs, val_docs


def train(docs: List[str], feature_dims: int, num_epochs: int):
    return Doc2Vec(docs, vector_size=feature_dims, window=2, min_count=1, workers=4, epochs=num_epochs)
