import argparse
import os
import sys

import requests
import jsonlines
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

from shared.utils import append_to_jsonl
from shared.global_constants import RES_DIR
from shared.utils import save_dict_to_json, tokenize_prune
from gnn.dataloading.build_graph import _load_text_and_labels
from preprocessing.stemming import create_stemming_map


def parse_arguments(args_to_parse):
    """ Parse CLI arguments """
    descr = "Download stemmed versions of word in vocab"
    parser = argparse.ArgumentParser(description=descr)

    general = parser.add_argument_group("General settings")
    general.add_argument(
        "--number-to-add",
        type=int,
        required=False,
        default=1000,
        help="Number of words from the vocab to download stemming data",
    )
    general.add_argument(
        "--count-threshold",
        type=int,
        required=False,
        default=1,
        help="Minimum word count for which stemming data will be downloaded",
    )
    general.add_argument(
        "--results-dir",
        type=str,
        default="gnn_results",
        help="Location of gnn results",
    )
    general.add_argument(
        "--input-data-dir",
        type=str,
        required=False,
        default="data",
        help="The name of the directory in which we have a our processed corpora (in a DataFrame)",
    )

    return parser.parse_args(args_to_parse)


def setup_dir(stemming_dir: str) -> str:
    os.makedirs(stemming_dir, exist_ok=True)
    stemming_download_path = os.path.join(stemming_dir, "stemming_results.jsonl")
    stemming_cleaned_path = os.path.join(stemming_dir, "stemming_cleaned.json")
    return stemming_download_path, stemming_cleaned_path


def create_vocab_counts(
    df_path: str, vocab_counts_path: str, text_column: str = "document_content", label_column: str = "document_type"
) -> None:
    if not os.path.isfile(df_path):
        raise FileNotFoundError(
            f"{df_path} could not be found.\
                Remember that you first need to generate the dataset using the `create_dataset` script"
        )
    document_list, labels = _load_text_and_labels(df_path, text_column, label_column)

    # Create a vocab list sorted by the frequency of each word
    print("Creating vocab with word counts...")
    cv = CountVectorizer(tokenizer=tokenize_prune)
    cv_fit = cv.fit_transform(document_list)
    word_list = cv.get_feature_names()
    count_list = cv_fit.toarray().sum(axis=0)
    word_counts = {word: count for word, count in zip(word_list, count_list)}
    vocab_sorted = {word: int(count) for word, count in Counter(word_counts).most_common()}
    save_dict_to_json(vocab_sorted, vocab_counts_path, sort_keys=False)


def load_vocab_counts(vocab_counts_path: str) -> dict:
    with open(vocab_counts_path, "r") as f:
        vocab_counts = json.load(f)
    return vocab_counts


def get_done_words(stemming_download_path: str) -> set:
    done_words = set()
    if os.path.exists(stemming_download_path):
        with jsonlines.open(stemming_download_path) as reader:
            for obj in reader:
                done_words.add(obj["word"])
    return done_words


def get_words_to_add(vocab_counts: dict, done_words: set, number_to_add: int, count_threshold: int) -> List[str]:
    words_above_threshold = [word for word, count in vocab_counts.items() if count >= count_threshold]
    words_to_add = [word for word in words_above_threshold if word not in done_words][:number_to_add]
    return words_above_threshold, words_to_add


def add_words(words_to_add: List[str], stemming_download_path: str) -> None:
    if len(words_to_add) == 0:
        print("All stemming data downloaded")
        return
    for word in tqdm(words_to_add):
        query_word(stemming_download_path, word)


def extract_stem(text: str) -> str:
    return text.split("[")[1].split("]")[0]


def query_word(stemming_download_path: str, word: str) -> None:
    query_data = {}
    query_data["word"] = word
    try:
        base_url = "http://77.240.23.241/dictionary"
        url = f"{base_url}/{word}/1"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        box = soup.find(class_="brown-box")
        if box is None:
            query_data["box_text"] = ""
            query_data["stem"] = ""
        else:
            query_data["box_text"] = box.text
            query_data["stem"] = extract_stem(box.text)
        query_data["status"] = 1
    except Exception as exception:
        query_data["exception"] = str(exception)
        query_data["status"] = 0
        print(f"Exception of type {str(exception)} for word {word}")
    append_to_jsonl(stemming_download_path, query_data)


def main(args):
    number_to_add = args.number_to_add
    count_threshold = args.count_threshold
    results_dir = os.path.join(RES_DIR, args.results_dir)
    vocab_counts_path = os.path.join(results_dir, "stemming", "vocab_counts.json")
    stemming_dir = os.path.join(results_dir, "stemming")
    df_path = os.path.join(RES_DIR, args.input_data_dir, "dataset.csv")

    os.makedirs(results_dir, exist_ok=True)
    stemming_download_path, stemming_cleaned_path = setup_dir(stemming_dir)

    if not os.path.isfile(vocab_counts_path):
        create_vocab_counts(df_path, vocab_counts_path)
    vocab_counts = load_vocab_counts(vocab_counts_path)

    done_words = get_done_words(stemming_download_path)

    words_above_threshold, words_to_add = get_words_to_add(vocab_counts, done_words, number_to_add, count_threshold)

    if len(words_to_add) > 0:
        print(f"{len(done_words)} words done out of {len(words_above_threshold)} words in vocab above count threshold")
        add_words(words_to_add, stemming_download_path)
    else:
        create_stemming_map(stemming_download_path, stemming_cleaned_path)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
