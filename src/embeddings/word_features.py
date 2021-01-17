from gensim.models import Word2Vec
import logging
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, List

from shared.utils import tokenize_prune_stem


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_word2vec(
    save_dir: str,
    document_list: List[str],
    stemming_map: Dict[str, str],
    num_epochs: int,
    embedding_dimension: int,
    training_regime: int,
) -> Word2Vec:
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
    model = Word2Vec(document_list, size=embedding_dimension, window=5, workers=4, sg=training_regime, min_count=1)
    model.save(os.path.join(save_dir, 'word2vec.model'))

    return model


def infer_word2vec_embeddings(model: Word2Vec, word_list: List[str]) -> np.ndarray:
    """
    NOTE: Inference is not deterministic therefore representations will vary between calls
    Returns a 2D array with shape (num_words, embedding_dimension)
    """
    print('Infering word embeddings..')
    return np.array([model[word] for word in word_list])
