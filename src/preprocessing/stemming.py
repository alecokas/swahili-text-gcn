import re
from typing import Dict, List

from shared.utils import read_jsonl, save_dict_to_json


def create_stemming_map(raw_path_name: str, cleaned_path_name: str) -> None:
    raw_salama_results = read_jsonl(raw_path_name)
    stemming_map = _generate_initial_map(raw_salama_results)
    stemming_map = _eliminate_single_repeated_char_words(stemming_map)
    stemming_map = _merge_repeated_letter_entries(stemming_map)
    # stemming_map = _merge_onomatopoeic_words(stemming_map)
    save_dict_to_json(stemming_map, cleaned_path_name)


def _generate_initial_map(raw_salama_results: List[Dict[str, str]]) -> Dict[str, str]:
    return {
        raw_result['word']: raw_result['stem'] if raw_result['stem'] != '' else raw_result['word']
        for raw_result in raw_salama_results
    }


def _eliminate_single_repeated_char_words(stemming_map: Dict[str, str]) -> Dict[str, str]:
    """ Remove words which are simply made up of a repeated character - such as `oooo` or `ff` """
    all_words = set(stemming_map.keys())
    regex = re.compile(r"^([a-z])\1+$")
    single_repeated_char_words = list(filter(regex.search, list(all_words)))

    for single_repeated_char_word in single_repeated_char_words:
        del stemming_map[single_repeated_char_word]

    return stemming_map

def _merge_repeated_letter_entries(stemming_map: Dict[str, str]) -> Dict[str, str]:
    """ TODO: For example merge the pairs: {`aisee`: `aisee`} and {`aiseee`: `aiseee`} """
    all_words = set(stemming_map.keys())
    regex = re.compile(r"^([a-z])\1\1+|([a-z])\2\2+$")
    words_with_repeated_chars = list(filter(regex.search, list(all_words)))

    print(words_with_repeated_chars)
    # TODO: roll back the repeated character
    return stemming_map


def _merge_onomatopoeic_words(stemming_map: Dict[str, str]) -> Dict[str, str]:
    """ TODO: Words with patterns, such as `hahaha` and `haha` should be merged """
    pass
