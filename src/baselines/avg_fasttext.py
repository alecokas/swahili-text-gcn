import fasttext.util
import numpy as np
import os
import shutil
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple

from shared.global_constants import RES_DIR
from shared.loaders import load_text_and_labels, save_categorical_labels
from shared.utils import read_json_as_dict, tokenize_prune_stem, write_to_meta, check_df_and_stemming_paths


def build_avg_fasttext_from_df(
    save_dir: str, df_path: str, stemming_map_path: str, text_column: str, label_column: str
) -> None:
    check_df_and_stemming_paths(df_path, stemming_map_path)
    stemming_map = read_json_as_dict(stemming_map_path)
    document_list, labels = load_text_and_labels(df_path, text_column, label_column)
    save_categorical_labels(save_dir, labels, as_numpy=True)

    # Tokenize
    cv = CountVectorizer(tokenizer=lambda text: tokenize_prune_stem(text, stemming_map=stemming_map))
    cv_tokenizer = cv.build_tokenizer()
    document_list = [cv_tokenizer(document) for document in document_list]
    # Load FastText and generate average embeddings
    ft_model = _load_pretrained_swahili_fasttext(RES_DIR)
    avg_ft_document_embeddings = _generate_avg_ft_document_embedding(ft_model, document_list)
    np.save(os.path.join(save_dir, 'ft-embeddings.npy'), avg_ft_document_embeddings)

    # Printouts
    num_docs = avg_ft_document_embeddings.shape[0]
    dims = avg_ft_document_embeddings.shape[1]
    print(f'{num_docs} documents have been embedded into {dims} dims')
    # Save meta-data to disk
    write_to_meta(
        data_meta_path=os.path.join(save_dir, 'meta.json'),
        key_val={
            'embedding_dims': dims,
            'num_docs': len(document_list),
        },
    )


def _load_pretrained_swahili_fasttext(save_location: str):
    """
    Model was trained using CBOW with position-weights, in dimension 300,
    with character n-grams of length 5, a window of size 5 and 10 negatives
    """
    model_name = 'cc.sw.300.bin'
    target_path_name = os.path.join(save_location, model_name)

    if not os.path.isfile(target_path_name):
        fasttext.util.download_model('sw', if_exists='ignore')
        shutil.move(model_name, target_path_name)
    return fasttext.load_model(target_path_name)


def _generate_avg_ft_document_embedding(ft_model, document_list: List[List[str]]) -> np.ndarray:
    avg_doc_embeddings = [
        np.mean(np.array(_generate_ft_embeddings(ft_model, document)), axis=0) for document in document_list
    ]
    return np.array(avg_doc_embeddings)


def _generate_ft_embeddings(ft_model, word_list: List[str]) -> List[np.ndarray]:
    return [ft_model.get_word_vector(word) for word in word_list]


def load_avg_fasttext(preproc_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    ft_embeddings = np.load(os.path.join(preproc_dir, 'ft-embeddings.npy'))
    labels = np.load(os.path.join(preproc_dir, 'labels.npy'))
    return ft_embeddings, labels
