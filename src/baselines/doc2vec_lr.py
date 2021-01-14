from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple

from shared.global_constants import RES_DIR
from shared.utils import read_json_as_dict, tokenize_prune_stem, write_to_meta, check_df_and_stemming_paths
from shared.loaders import load_text_and_labels, save_categorical_labels


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    filename=os.path.join(RES_DIR, 'doc2vec-training-log'),
)


def build_doc2vec_from_df(
    save_dir: str,
    df_path: str,
    stemming_map_path: str,
    text_column: str,
    label_column: str,
    training_regime: int,
    embedding_dimension: int,
    num_epochs: int,
) -> None:
    check_df_and_stemming_paths(df_path, stemming_map_path)
    stemming_map = read_json_as_dict(stemming_map_path)
    document_list, labels = load_text_and_labels(df_path, text_column, label_column)
    save_categorical_labels(save_dir, labels, as_numpy=True)

    # Tokenize
    cv = CountVectorizer(tokenizer=lambda text: tokenize_prune_stem(text, stemming_map=stemming_map))
    cv_tokenizer = cv.build_tokenizer()
    document_list = [cv_tokenizer(document) for document in document_list]

    # Convert to TaggedDocument and train
    print('Training Doc2Vec...')
    tagged_document_list = [TaggedDocument(doc, [i]) for i, doc in enumerate(document_list)]
    doc2vec_model = _train_doc2vec(
        docs=tagged_document_list,
        feature_dims=embedding_dimension,
        num_epochs=num_epochs,
        training_regime=training_regime,
    )

    doc_vecs = _infer_document_embeddings(doc2vec_model, document_list)
    print(f'{doc_vecs.shape[0]} documents, each with {doc_vecs.shape[1]} features')
    assert doc_vecs.shape[0] == len(
        tagged_document_list
    ), f'Expected {doc_vecs.shape[0]} == {len(tagged_document_list)}'
    np.save(os.path.join(save_dir, 'doc2vec-embeddings.npy'), doc_vecs)

    # Save and meta-data to disk
    write_to_meta(
        data_meta_path=os.path.join(save_dir, 'meta.json'),
        key_val={
            'training_regime': 'PV-DM' if training_regime == 1 else 'PV-DBOW',
            'num_docs': len(document_list),
            'vector_size': embedding_dimension,
        },
    )


def _train_doc2vec(docs: List[TaggedDocument], feature_dims: int, num_epochs: int, training_regime: int) -> Doc2Vec:
    model = Doc2Vec(vector_size=feature_dims, window=5, min_count=2, workers=4, epochs=num_epochs, dm=training_regime)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=num_epochs)
    return model


def _infer_document_embeddings(model: Doc2Vec, doc_list: List[List[str]]) -> np.ndarray:
    """
    NOTE: Inference is not deterministic therefore representations will vary between calls
    Returns a 2D array with shape (num_docs, embedding_dimension)
    """
    print('Infering document embeddings..')
    return np.array([model.infer_vector(doc) for doc in doc_list])


def load_doc2vec(preproc_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    doc_vecs = np.load(os.path.join(preproc_dir, 'doc2vec-embeddings.npy'))
    labels = np.load(os.path.join(preproc_dir, 'labels.npy'))
    return doc_vecs, labels
