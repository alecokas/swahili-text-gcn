from collections import defaultdict
from math import log
import numpy as np
import os
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack, identity, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import torch
from typing import Dict, List, Set, Tuple, Optional

from shared.utils import save_dict_to_json, read_json_as_dict, tokenize_prune_stem, write_to_meta


def build_graph_from_df(
    graph_dir: str, df_path: str, stemming_map_path: str, text_column: str, label_column: str, window_size: int
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
    document_list, labels = _load_text_and_labels(df_path, text_column, label_column)
    label_to_cat = {label: cat for cat, label in enumerate(set(labels))}
    catagorical_labels = torch.LongTensor([label_to_cat[label] for label in labels])

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
    windows = _create_window_contexts(document_list, window_size)

    word_occurence_count_map = _create_word_occurence_count_map(windows)
    word_pair_occurence_count_map = _create_word_pair_occurence_count_map(windows)

    word_cooccurrences_list = _word_cooccurrences(
        words_list=list(token_to_int_vocab_map.keys()),
        word_occurence_count_map=word_occurence_count_map,
        word_pair_occurence_count_map=word_pair_occurence_count_map,
        num_windows=len(windows),
    )
    print(f'There are {len(word_cooccurrences_list)} word-word co-occurence weights')

    adjacency = _merge_into_adjacency(tf_ids, word_cooccurrences_list, token_to_int_vocab_map)
    print(f'The adjacency has size: {adjacency.shape}')

    input_features = torch.eye(adjacency.shape[0]).to_sparse()
    print(f'Input features size: {input_features.shape}')

    # Save graph, labels, and meta-data to disk
    write_to_meta(
        data_meta_path=os.path.join(graph_dir, 'meta.json'),
        key_val={
            'vocab_size': len(token_to_int_vocab_map),
            'num_docs': len(document_list),
            'num_wondows': len(windows),
        },
    )
    save_dict_to_json(label_to_cat, os.path.join(graph_dir, 'label_map.json'))
    save_dict_to_json(token_to_int_vocab_map, os.path.join(graph_dir, 'vocab_map.json'))
    torch.save(catagorical_labels, os.path.join(graph_dir, 'labels.pt'))
    torch.save(input_features, os.path.join(graph_dir, 'input_features.pt'))
    save_npz(os.path.join(graph_dir, 'adjacency.npz'), adjacency)


def _load_text_and_labels(df_path: str, text_column: str, label_column: str) -> Tuple[List[List[str]], List[str]]:
    """
    Load the CSV from file, extract the column we want along with the corresponding labels.
    """
    dataset_df = pd.read_csv(df_path, sep=';')
    document_content = dataset_df[text_column].values.tolist()
    labels = dataset_df[label_column].values.tolist()
    assert len(labels) == len(document_content), 'The number of labels and docs should match'
    return document_content, labels


def _create_window_contexts(doc_list: List[str], window_size: int, stemming_map: Dict[str, str]) -> List[Set[str]]:
    """
    NOTE: not all windows will be the same size.
    Specifically windows taken from documents which are shorter than the window size.
    """
    windows = []
    for doc in doc_list:
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
    stacked_node_interactions = vstack([word_coocurrences, tf_ids])
    zero_csr = csr_matrix(([], ([], [])), shape=(stacked_node_interactions.shape[0], tf_ids.shape[0]))
    adj = hstack([stacked_node_interactions, zero_csr]) + identity(stacked_node_interactions.shape[0])

    assert adj.shape == (
        len(token_to_int_vocab_map) + tf_ids.shape[0],
        len(token_to_int_vocab_map) + tf_ids.shape[0],
    ), "Expected {} == {}".format(
        adj.shape, (len(token_to_int_vocab_map) + tf_ids.shape[0], len(token_to_int_vocab_map) + tf_ids.shape[0])
    )
    return adj
