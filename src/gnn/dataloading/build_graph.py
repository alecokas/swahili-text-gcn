from collections import defaultdict
from math import log
import numpy as np
import os
from scipy.sparse import csr_matrix, hstack, vstack, identity, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import torch
from typing import Dict, List, Set, Tuple, Optional

from embeddings.doc_features import get_doc2vec_embeddngs
from embeddings.word_features import train_word2vec, infer_word2vec_embeddings
from shared.loaders import load_text_and_labels, save_categorical_labels
from shared.utils import (
    save_dict_to_json,
    read_json_as_dict,
    tokenize_prune_stem,
    write_to_meta,
    check_df_and_stemming_paths,
)


def build_graph_from_df(
    graph_dir: str,
    df_path: str,
    stemming_map_path: str,
    input_feature_type: str,
    text_column: str,
    label_column: str,
    window_size: int,
    wv_sg: int = 1,
    pv_dm: int = 1,
) -> None:
    check_df_and_stemming_paths(df_path, stemming_map_path)
    stemming_map = read_json_as_dict(stemming_map_path)
    document_list, labels = load_text_and_labels(df_path, text_column, label_column)
    save_categorical_labels(graph_dir, labels)

    # Obtain TF-IDF for word-document weights: TODO: strip_accents='unicode'
    print('TF-IDF...')
    vectoriser = TfidfVectorizer(tokenizer=lambda text: tokenize_prune_stem(text, stemming_map=stemming_map))
    tf_ids = vectoriser.fit_transform(document_list)
    print(tf_ids.shape)
    token_to_int_vocab_map = vectoriser.vocabulary_
    print(f'There are {len(document_list)} documents in our corpus')
    print(f'Our vocabulary has {len(token_to_int_vocab_map)} words in it')
    print(tf_ids.shape)
    save_dict_to_json(token_to_int_vocab_map, os.path.join(graph_dir, 'vocab_map.json'))

    # Obtain word co-occurence statistics (PMI) for word-word weights
    print('Word Co-ocurrences...')
    windows = []
    for document in tqdm(document_list, desc='Generating all windows: '):
        windows.extend(_create_window_contexts(document, window_size, stemming_map))
    word_occurence_count_map = _create_word_occurence_count_map(windows)
    word_pair_occurence_count_map = _create_word_pair_occurence_count_map(windows)

    # Save the number of windows and delete the list to save RAM
    num_windows = len(windows)
    del windows

    word_list = ordered_word_list(token_to_int_vocab_map)
    assert _check_order(
        word_list, token_to_int_vocab_map
    ), 'word_list is not consistent with the token indices in token_to_int_vocab_map'

    word_cooccurrences_list = _word_cooccurrences(
        words_list=word_list,
        word_occurence_count_map=word_occurence_count_map,
        word_pair_occurence_count_map=word_pair_occurence_count_map,
        num_windows=num_windows,
    )
    print(f'There are {len(word_cooccurrences_list)} word-word co-occurence weights')

    adjacency = _merge_into_adjacency(tf_ids, word_cooccurrences_list, token_to_int_vocab_map)
    save_npz(os.path.join(graph_dir, 'adjacency.npz'), adjacency)

    # Store adjacency shape and delete adjacency to save RAM
    adjacency_shape = adjacency.shape
    print(f'The adjacency has size: {adjacency_shape}')
    del adjacency

    if input_feature_type == 'one-hot':
        input_features = torch.eye(adjacency_shape[0]).to_sparse()
    elif input_feature_type == 'text2vec':
        # For now, keep these doc2vec settings constant
        input_doc_features = get_doc2vec_embeddngs(
            save_dir=graph_dir,
            document_list=document_list,
            stemming_map=stemming_map,
            num_epochs=20,
            vector_size=300,
            training_regime=pv_dm,
        )
        word2vec_model = train_word2vec(
            save_dir=graph_dir,
            document_list=document_list,
            stemming_map=stemming_map,
            num_epochs=20,
            embedding_dimension=300,
            training_regime=wv_sg,
        )
        input_word_features = infer_word2vec_embeddings(word2vec_model, word_list)
        del word2vec_model
        # The order of concatenation is important. It must match the order in adjacency.
        input_features = torch.FloatTensor(
            np.concatenate([input_word_features, input_doc_features], axis=0)
        ).to_sparse()
        print(f'input_word_features.shape: {input_word_features.shape}')
        print(f'input_doc_features.shape: {input_doc_features.shape}')
    else:
        raise TypeError(f'{input_feature_type} is not a valid input feature type')

    torch.save(input_features, os.path.join(graph_dir, 'input_features.pt'))
    print(f'Input features size: {input_features.shape}')

    # Save graph, labels, and meta-data to disk
    write_to_meta(
        data_meta_path=os.path.join(graph_dir, 'meta.json'),
        key_val={
            'vocab_size': len(token_to_int_vocab_map),
            'num_docs': len(document_list),
            'num_windows': num_windows,
            'window_size': window_size,
            'num_word_coocurences': len(word_cooccurrences_list),
        },
    )


def _create_window_contexts(doc: str, window_size: int, stemming_map: Dict[str, str]) -> List[Set[str]]:
    """
    NOTE: not all windows will be the same size.
    Specifically windows taken from documents which are shorter than the window size.
    """
    windows = []
    words = tokenize_prune_stem(doc, stemming_map=stemming_map)
    if len(words) <= window_size:
        windows.append(set(words))
    else:
        for i in range(len(words) - window_size + 1):
            windows.append(set(words[i : i + window_size]))
    return windows


def _word_cooccurrences(
    words_list: List[str],
    word_occurence_count_map: Dict[str, int],
    word_pair_occurence_count_map: Dict[str, int],
    num_windows: int,
) -> List[Tuple[str, str, float]]:
    word_cooccurrences_list = []
    for i, word_i in tqdm(enumerate(words_list[:-1]), desc='Creating PMI weights: '):
        for j in range(i + 1, len(words_list)):
            pmi = _pointwise_mi(
                word_i=word_i,
                word_j=words_list[j],
                word_occurence_count_map=word_occurence_count_map,
                word_pair_occurence_count_map=word_pair_occurence_count_map,
                num_windows=num_windows,
            )
            if pmi is not None and pmi > 0:
                word_cooccurrences_list.append((word_i, words_list[j], pmi))
    return word_cooccurrences_list


def _create_word_occurence_count_map(windows: List[Set[str]]) -> Dict[str, int]:
    """ Produce a dictionary which indicates the number of sliding windows which contain a particular word """
    word_occurence_count_map = defaultdict(int)
    for window in tqdm(windows, desc='Creating word_occurence_count_map: '):
        for word in list(window):
            word_occurence_count_map[word] += 1
    return dict(word_occurence_count_map)


def _create_word_pair_occurence_count_map(windows: List[Set[str]]) -> Dict[str, int]:
    """ Produce a dictionary which indicates the number of sliding windows which contain a particular pair of words """
    word_pair_occurence_count_map = defaultdict(int)
    for window in tqdm(windows, desc='Creating create_word_pair_occurence_count_map: '):
        window_list = list(window)
        for i, word_i in enumerate(window_list[:-1]):
            for word_j in window_list[i + 1 : len(window_list)]:
                if word_i != word_j:
                    word_pair_occurence_count_map[f'{word_i},{word_j}'] += 1
                    word_pair_occurence_count_map[f'{word_j},{word_i}'] += 1
    return dict(word_pair_occurence_count_map)


def _pointwise_mi(
    word_i: str,
    word_j: str,
    word_occurence_count_map: Dict[str, int],
    word_pair_occurence_count_map: Dict[str, int],
    num_windows: int,
) -> Optional[float]:
    """
    Calculate the pointwise mutual information between words i and j.
    If joint_prob we avoid taking the log() and just return None which we can ignore.
    """
    if f'{word_i},{word_j}' not in word_pair_occurence_count_map:
        return None

    joint_prob = word_pair_occurence_count_map[f'{word_i},{word_j}'] / num_windows
    marginal_i_prob = word_occurence_count_map[word_i] / num_windows
    marginal_j_prob = word_occurence_count_map[word_j] / num_windows

    assert marginal_i_prob != 0, f'No instances of {word_i} found - this should never happen'
    assert marginal_j_prob != 0, f'No instances of {word_j} found - this should never happen'

    return log(joint_prob / (marginal_i_prob * marginal_j_prob)) if joint_prob > 0 else None


def _merge_into_adjacency(
    tf_ids: csr_matrix, word_cooccurrences_list: List[Tuple[str, str, float]], token_to_int_vocab_map: Dict[str, int]
) -> csr_matrix:
    """
    Merge the word co-occurence information together with the tf-idf information to create an adjacency matrix
    where,
        (0, 0) to (|vocab|, |vocab|) - indices describe the word-word interactions
    and,
        (|vocal|, |vocab|) to (|vocab| + #Docs, |vocab|) - indices describe the word-document interactions.
    """
    word_co_row = np.array(
        [token_to_int_vocab_map[word_cooccurrence[0]] for word_cooccurrence in word_cooccurrences_list]
    )
    word_co_col = np.array(
        [token_to_int_vocab_map[word_cooccurrence[1]] for word_cooccurrence in word_cooccurrences_list]
    )
    word_co_data = np.array([word_cooccurrence[2] for word_cooccurrence in word_cooccurrences_list])
    word_coocurrences = csr_matrix(
        (word_co_data, (word_co_row, word_co_col)), shape=(len(token_to_int_vocab_map), len(token_to_int_vocab_map))
    )
    # Stack word co-occurences ontop of TF-IDF (Left hand side of adjacency)
    adj_lhs = vstack([word_coocurrences, tf_ids])
    # Empty (zeros) for doc-doc interactions
    zero_csr = csr_matrix(([], ([], [])), shape=(tf_ids.shape[0], tf_ids.shape[0]))
    # Mirror TF-IDFs and stack ontop of doc-doc interactions to create right hand side of the adjacency
    adj_rhs = vstack([tf_ids.transpose(), zero_csr])
    # Stack side-by-side
    adj = hstack([adj_lhs, adj_rhs]) + identity(adj_lhs.shape[0])

    assert adj.shape == (
        len(token_to_int_vocab_map) + tf_ids.shape[0],
        len(token_to_int_vocab_map) + tf_ids.shape[0],
    ), "Expected {} == {}".format(
        adj.shape, (len(token_to_int_vocab_map) + tf_ids.shape[0], len(token_to_int_vocab_map) + tf_ids.shape[0])
    )
    return adj


def _check_order(word_list: List[str], token_to_int_vocab_map: Dict[str, int]) -> bool:
    for index, word in enumerate(word_list):
        if token_to_int_vocab_map[word] != index:
            return False
    return True


def ordered_word_list(token_to_int_vocab_map: Dict[str, int]) -> List[str]:
    word_list = [None] * len(token_to_int_vocab_map)
    for word, idx in token_to_int_vocab_map.items():
        word_list[idx] = word
    if None in word_list:
        raise Exception('There is a `None` element in the list')
    return word_list
