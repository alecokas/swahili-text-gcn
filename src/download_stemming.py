import argparse
import os
import sys

import requests
import jsonlines
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import List

from shared.utils import append_to_jsonl
from shared.global_constants import RES_DIR


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
        "--results-dir",
        type=str,
        default="gnn_results",
        help="Location of gnn results",
    )

    general.add_argument(
        "--stemming-dir",
        type=str,
        default="stemming",
        help="The name of the subdirectory where we should save stemming data",
    )

    return parser.parse_args(args_to_parse)


def setup_dir(stemming_dir: str) -> str:
    os.makedirs(stemming_dir, exist_ok=True)
    save_path = os.path.join(stemming_dir, "stemming_results.jsonl")
    return save_path


def load_vocab_counts(vocab_counts_path: str) -> dict:
    with open(vocab_counts_path, "r") as f:
        vocab_count = json.load(f)
    return vocab_count


def get_done_words(save_path: str) -> set:
    done_words = set()
    if os.path.exists(save_path):
        with jsonlines.open(save_path) as reader:
            for obj in reader:
                done_words.add(obj["word"])
    return done_words


def get_words_to_add(vocab_count: dict, done_words: set, number_to_add: int) -> List[str]:
    words_to_add = [word for word in vocab_count.keys() if word not in done_words][
        :number_to_add
    ]
    return words_to_add


def add_words(words_to_add: List[str], save_path: str):
    for word in tqdm(words_to_add):
        query_word(save_path, word)


def extract_stem(text: str):
    return text.split("[")[1].split("]")[0]


def query_word(save_path: str, word: str) -> None:
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
        query_data["exception"] = 0
        query_data["status"] = 0
    append_to_jsonl(save_path, query_data)


def main(args):
    number_to_add = args.number_to_add
    results_dir = os.path.join(RES_DIR, args.results_dir)
    vocab_counts_path = os.path.join(results_dir, "graph", "vocab_counts.json")
    stemming_dir = os.path.join(RES_DIR, args.stemming_dir)

    save_path = setup_dir(stemming_dir)

    vocab_count = load_vocab_counts(vocab_counts_path)

    done_words = get_done_words(save_path)

    print(f"{len(done_words)} words done out of {len(vocab_count)} words in vocab")

    words_to_add = get_words_to_add(vocab_count, done_words, number_to_add)

    add_words(words_to_add, save_path)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
