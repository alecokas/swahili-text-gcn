from datetime import datetime
import json
import jsonlines
from nltk import word_tokenize
import os
from typing import Any, Dict, List, Optional

from shared.global_constants import STOP_WORDS


def rm_file(file_name: str) -> None:
    try:
        os.remove(file_name)
    except OSError:
        pass


def save_dict_to_json(dict_map: Dict[str, int], file_path: str, sort_keys: bool = True):
    with open(file_path, 'w') as fp:
        json.dump(dict_map, fp, sort_keys=sort_keys, indent=4)


def read_json_as_dict(path: str) -> Dict[str, Any]:
    with open(path) as json_file:
        return json.load(json_file)


def tokenize_prune_stem(text: str, stemming_map: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Use NLTK word tokenisation and clean our text, and (if passed to the function) use stemming map to stem all words.
    NOTE: If `word` is not in `stemming_map`, we do not include it
    """
    return [
        stemming_map[word] if stemming_map is not None else word
        for word in word_tokenize(text)
        if len(word) > 1 and word.isalpha() and word not in STOP_WORDS and word in stemming_map
    ]


def tokenize_prune(text: str) -> List[str]:
    """ Use NLTK word tokenisation and clean our text """
    return [
        word
        for word in word_tokenize(text)
        if len(word) > 1 and word.isalpha() and word not in STOP_WORDS
    ]


def write_list_to_file(list_of_strings: List[str], target_file: str) -> None:
    """ Write a list of strings to a target file """
    with open(target_file, 'w') as tgt_file:
        tgt_file.write('\n'.join(list_of_strings))


def append_to_jsonl(path: str, dict_to_append: Dict[str, Any]):
    with jsonlines.open(path, mode='a') as writer:
        writer.write(dict_to_append)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    json_list = []
    with jsonlines.open(path) as reader:
        for json_obj in reader:
            json_list.append(json_obj)
    return json_list


def write_to_meta(data_meta_path: str, key_val: Dict[str, Any]) -> None:
    """ Write the key-value pair to a json meta file """
    if os.path.isfile(data_meta_path):
        with open(data_meta_path, 'r') as meta_file:
            meta_data = json.load(meta_file)
    else:
        meta_data = {}

    meta_data.update(key_val)

    with open(data_meta_path, 'w') as meta_file:
        json.dump(meta_data, meta_file)


def save_cli_options(options, save_dir: str) -> None:
    """ Save all options to JSON file.
        Arguments:
            options: A Namespace object from argparse
            save_dir: String location to save the options
    """
    opt_dict = {}
    for option in vars(options):
        opt_dict[option] = getattr(options, option)

    os.makedirs(save_dir, exist_ok=True)
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y-%H:%M:%S")
    opts_file_path = os.path.join(save_dir, f"opts-{dt_string}.json")
    with open(opts_file_path, "w") as opt_file:
        json.dump(opt_dict, opt_file)
