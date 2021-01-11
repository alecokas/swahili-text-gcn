import numpy as np
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple

from shared.utils import save_dict_to_json, read_json_as_dict, tokenize_prune_stem, write_to_meta
from shared.loaders import load_text_and_labels, save_categorical_labels


def build_tfidf_from_df(
    save_dir: str, df_path: str, stemming_map_path: str, text_column: str, label_column: str
) -> None:
    if not os.path.isfile(df_path):
        raise FileNotFoundError(
            f'{df_path} could not be found.\
                Remember that you first need to generate the dataset using the `create_dataset` script'
        )
    if not os.path.isfile(stemming_map_path):
        raise FileNotFoundError(
            f'{stemming_map_path} could not be found.\
                Remember that you need to first generate a stemming map using the `download_stemming` script'
        )
    stemming_map = read_json_as_dict(stemming_map_path)
    document_list, labels = load_text_and_labels(df_path, text_column, label_column)
    save_categorical_labels(save_dir, labels, as_numpy=True)

    # Obtain TF-IDF for word-document weights: TODO: strip_accents='unicode'
    print('TF-IDF...')
    vectoriser = TfidfVectorizer(tokenizer=lambda text: tokenize_prune_stem(text, stemming_map=stemming_map))
    tf_idfs = vectoriser.fit_transform(document_list)
    save_npz(os.path.join(save_dir, 'tf-idf.npz'), tf_idfs)

    token_to_int_vocab_map = vectoriser.vocabulary_
    print(f'There are {len(document_list)} documents in our corpus')
    print(f'Our vocabulary has {len(token_to_int_vocab_map)} words in it')

    save_dict_to_json(token_to_int_vocab_map, os.path.join(save_dir, 'vocab_map.json'))

    # Save and meta-data to disk
    write_to_meta(
        data_meta_path=os.path.join(save_dir, 'meta.json'),
        key_val={
            'vocab_size': len(token_to_int_vocab_map),
            'num_docs': len(document_list),
        },
    )


def load_tfidf(preproc_dir: str) -> Tuple[csr_matrix, np.ndarray]:
    tf_idf = load_npz(os.path.join(preproc_dir, 'tf-idf.npz'))
    labels = np.load(os.path.join(preproc_dir, 'labels.npy'))
    return tf_idf, labels
