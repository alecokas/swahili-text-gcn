import argparse
from collections import Counter
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
import sys
from typing import Dict, Optional

from shared.global_constants import RES_DIR
from shared.loaders import load_text_and_labels
from preprocessing.stemming import (
    create_stemming_map,
    remove_stemming_entries_below_count_threshold,
    add_words_to_map,
    get_new_words_to_add,
    get_all_completed_words,
)
from shared.utils import (
    save_dict_to_json,
    read_json_as_dict,
    tokenize_prune,
    tokenize_prune_stem,
    save_cli_options,
)


def parse_arguments(args_to_parse):
    """ Parse CLI arguments """
    descr = "Download stemmed versions of word in vocab and apply cleaning / processing "
    parser = argparse.ArgumentParser(description=descr)

    general = parser.add_argument_group("General settings")
    general.add_argument(
        "--number-to-add",
        type=int,
        required=False,
        default=1000,
        help="Number of words from the vocab to download stemming data",
    )
    # NOTE: For the --count-threshold option: If the raw stemming data has already been downloaded
    # with a more lenient threshold, the stemming process will compensate for it in the cleaning stage
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
    df_path: str,
    vocab_counts_path: str,
    excl_word_count_lower_than: int = 1,
    stemming_map: Optional[Dict[str, str]] = None,
    text_column: str = "document_content",
    label_column: str = "document_type",
) -> None:
    if not os.path.isfile(df_path):
        raise FileNotFoundError(
            f"{df_path} could not be found.\
                Remember that you first need to generate the dataset using the `create_dataset` script"
        )
    document_list, labels = load_text_and_labels(df_path, text_column, label_column)

    # Create a vocab list sorted by the frequency of each word
    print("Creating vocab with word counts...")
    if stemming_map is None:
        cv = CountVectorizer(tokenizer=tokenize_prune, min_df=excl_word_count_lower_than)
    else:
        cv = CountVectorizer(
            tokenizer=lambda text: tokenize_prune_stem(text, stemming_map=stemming_map),
            min_df=excl_word_count_lower_than,
        )
    cv_fit = cv.fit_transform(document_list)
    word_list = cv.get_feature_names()
    # Convert to uint16 to keep the memory requirements down, otherwise approx. 34.3 GiB for Zenodo
    count_list = cv_fit.astype(dtype='uint16').toarray().sum(axis=0)
    word_counts = {word: count for word, count in zip(word_list, count_list)}
    vocab_sorted = {word: int(count) for word, count in Counter(word_counts).most_common()}
    save_dict_to_json(vocab_sorted, vocab_counts_path, sort_keys=False)


def load_vocab_counts(vocab_counts_path: str) -> dict:
    with open(vocab_counts_path, "r") as f:
        vocab_counts = json.load(f)
    return vocab_counts


def main(args):
    """ Entry point for generating a clean stemming map """
    results_dir = os.path.join(RES_DIR, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    vocab_counts_path = os.path.join(results_dir, "stemming", "vocab_counts.json")
    stemming_dir = os.path.join(results_dir, "stemming")
    df_path = os.path.join(RES_DIR, args.input_data_dir, "dataset.csv")
    stemming_download_path, stemming_cleaned_path = setup_dir(stemming_dir)

    save_cli_options(args, results_dir)

    if not os.path.isfile(vocab_counts_path):
        create_vocab_counts(df_path=df_path, vocab_counts_path=vocab_counts_path)
    vocab_counts = load_vocab_counts(vocab_counts_path)

    done_words = get_all_completed_words(stemming_download_path)

    words_above_threshold, words_to_add = get_new_words_to_add(
        vocab_counts, done_words, args.number_to_add, args.count_threshold
    )

    print(f'Number of words to add: {len(words_to_add)}')
    if len(words_to_add) > 0:
        print(f"{len(done_words)} words done out of {len(words_above_threshold)} words in vocab above count threshold")
        add_words_to_map(words_to_add, stemming_download_path)
    else:
        # Convert the raw data into a cleaned stemming mapp
        create_stemming_map(stemming_download_path, stemming_cleaned_path)
        # Get the vocab count after the stemming and cleaning process has taken place
        cleaned_vocab_path = os.path.join(results_dir, "stemming", "cleaned_vocab_counts.json")
        print('First pass cleaned and stemmed vocab_counts...')
        stemming_map = read_json_as_dict(stemming_cleaned_path)
        create_vocab_counts(
            df_path=df_path,
            vocab_counts_path=cleaned_vocab_path,
            excl_word_count_lower_than=args.count_threshold,
            stemming_map=stemming_map,
        )
        print('Second pass at cleaning the stemming vocab_counts...')
        remove_stemming_entries_below_count_threshold(stemming_cleaned_path, cleaned_vocab_path)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
