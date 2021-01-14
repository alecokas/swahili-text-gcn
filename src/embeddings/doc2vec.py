from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import numpy as np
import os
import pandas as pd
import torch
from typing import List, Tuple


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def _read_and_format_tagged_docs(
    df_path: str, text_column: str, label_column: str
) -> Tuple[List[TaggedDocument], List[str]]:
    dataset_df = pd.read_csv(df_path, sep=';')
    docs_list = dataset_df[text_column].values.tolist()
    docs_list = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs_list)]
    labels = dataset_df[label_column].values.tolist()
    assert len(labels) == len(docs_list), 'The number of labels and docs should match'
    return docs_list, labels


def separate_into_subsets(
    docs: List[TaggedDocument], input_dir: str
) -> Tuple[List[TaggedDocument], List[TaggedDocument]]:
    """ Separate the docs into the predefined train and val lists """
    train_indices = torch.load(os.path.join(input_dir, 'train-indices.pt')).tolist()
    val_indices = torch.load(os.path.join(input_dir, 'val-indices.pt')).tolist()

    train_docs = [docs[idx] for idx in train_indices]
    val_docs = [docs[idx] for idx in val_indices]

    return train_docs, val_docs


def train(docs: List[str], feature_dims: int, num_epochs: int) -> Doc2Vec:
    model = Doc2Vec(vector_size=feature_dims, min_count=2, workers=4, epochs=num_epochs)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=num_epochs)
    return model


def save_for_inference(model: Doc2Vec, path_name: str) -> None:
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model.save(path_name)


def infer_document_embeddings(model_path: str, doc_list: List[List[str]]) -> List[np.ndarray]:
    """ NOTE: Inference is not deterministic therefore representations will vary between calls """
    model = Doc2Vec.load(model_path)
    # TODO: concat list elements sos that it is a 2D array
    return [model.infer_vector(doc) for doc in doc_list]
