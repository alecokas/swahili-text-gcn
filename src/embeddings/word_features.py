from gensim.models.word2vec import Word2Vec
import logging
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, List

from shared.utils import tokenize_prune_stem


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_word2vec_embeddngs(
    save_dir: str,
    document_list: List[str],
    word_list: List[str],
    stemming_map: Dict[str, str],
    num_epochs: int,
    embedding_dimension: int,
    training_regime: int,
) -> np.ndarray:
    """Training regime:
    - 1 for skip-gram;
    - otherwise (0) CBOW.
    """
    # Tokenize
    cv = CountVectorizer(tokenizer=lambda text: tokenize_prune_stem(text, stemming_map=stemming_map))
    cv_tokenizer = cv.build_tokenizer()
    document_list = [cv_tokenizer(document) for document in document_list]

    # Convert to TaggedDocument and train
    print('Training Word2Vec...')
    word2vec_model = _train_word2vec(
        docs=document_list,
        feature_dims=embedding_dimension,
        num_epochs=num_epochs,
        training_regime=training_regime,
    )
    _save_for_inference(word2vec_model, os.path.join(save_dir, 'word2vec.model'))

    return _infer_word_embeddings(word2vec_model, word_list)


def _train_word2vec(docs: List[str], feature_dims: int, num_epochs: int, training_regime: int) -> Word2Vec:
    model = Word2Vec(vector_size=feature_dims, window=5, workers=4, num_epochs=num_epochs, sg=training_regime)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=num_epochs)


def _infer_word_embeddings(model: Word2Vec, word_list: List[str]) -> np.ndarray:
    """
    NOTE: Inference is not deterministic therefore representations will vary between calls
    Returns a 2D array with shape (num_words, embedding_dimension)
    """
    print('Infering word embeddings..')
    return np.array([model.infer_vector(word) for word in word_list])


def _save_for_inference(model: Word2Vec, path_name: str) -> None:
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model.save(path_name)
