import errno
import json
import jsonlines
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import os
import torch
from typing import Any, Dict, List


stemmer = PorterStemmer()


def mkdir(directory: str) -> None:
    """ Make directory if it does not exist """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def rm_file(file_name: str) -> None:
    try:
        os.remove(file_name)
    except OSError:
        pass


def tokenize_and_stem(text: str) -> List[str]:
    """ Use NLTK word tokenisation and Porter stemmer. Also exclude any singe character words """
    tokens = [word for word in word_tokenize(text) if len(word) > 1]
    return [stemmer.stem(item) for item in tokens]


def write_list_to_file(list_of_strings: List[str], target_file: str) -> None:
    """ Write a list of strings to a target file """
    with open(target_file, 'w') as tgt_file:
        tgt_file.write('\n'.join(list_of_strings))


def append_to_jsonl(path: str, dict_to_append: Dict[str, Any]):
    with jsonlines.open(path, mode='a') as writer:
        writer.write(dict_to_append)


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


def save_dict_to_json(dict_map: Dict[str, int], file_path: str):
    with open(file_path, 'w') as fp:
        json.dump(dict_map, fp, sort_keys=True, indent=4)


def get_device(use_gpu: bool) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def remove_previous_best(directory: str, current_step: int) -> None:
    for item in os.listdir(directory):
        if f'model-{current_step}' not in item:
            rm_file(os.path.join(directory, item))
