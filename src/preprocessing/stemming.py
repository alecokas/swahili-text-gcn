import re
from typing import Dict, List

from preprocessing.text_stripper import ignore_non_ascii
from shared.utils import read_jsonl, save_dict_to_json


def create_stemming_map(raw_path_name: str, cleaned_path_name: str) -> None:
    raw_salama_results = read_jsonl(raw_path_name)
    stemming_map = _generate_initial_map(raw_salama_results)
    stemming_map = _ignore_non_ascii_entries(stemming_map)
    stemming_map = _eliminate_single_repeated_char_words(stemming_map)
    stemming_map = _merge_laughs_words(stemming_map)
    stemming_map = _merge_onomatopoeic_words(stemming_map)
    save_dict_to_json(stemming_map, cleaned_path_name)


def _generate_initial_map(raw_salama_results: List[Dict[str, str]], max_word_length: int = 30) -> Dict[str, str]:
    """ Create map - but also exclude all words longer than a threshold of characters. Assume these are an error """
    return {
        raw_result['word']: raw_result['stem'] if raw_result['stem'] != '' else raw_result['word']
        for raw_result in raw_salama_results
        if len(raw_result['word']) <= max_word_length
    }


def _ignore_non_ascii_entries(stemming_map: Dict[str, str]) -> Dict[str, str]:
    return {ignore_non_ascii(key): ignore_non_ascii(val) for key, val in stemming_map.items()}


def _eliminate_single_repeated_char_words(stemming_map: Dict[str, str]) -> Dict[str, str]:
    """ Remove words which are simply made up of a repeated character - such as `oooo` or `ff` """
    all_words = set(stemming_map.keys())
    regex = re.compile(r"^([a-z])\1+$")
    single_repeated_char_words = list(filter(regex.search, list(all_words)))

    for single_repeated_char_word in single_repeated_char_words:
        del stemming_map[single_repeated_char_word]

    return stemming_map


def _merge_onomatopoeic_words(stemming_map: Dict[str, str]) -> Dict[str, str]:
    """ Set eh's, ah, and ohs to the keyword onomatopoeia"""
    all_words = set(stemming_map.keys())
    regex = re.compile(r"^(a)\1+h+$|^a+(h)\2+$|^(e)\3+h+$|^e+(h)\4+$|^(o)\5+h+$|^o+(h)\6+$")
    onomatopoeic_words = list(filter(regex.search, list(all_words)))

    for onomatopoeic_word in onomatopoeic_words:
        stemming_map[onomatopoeic_word] = 'onomatopoeia'
    return stemming_map


def _merge_laughs_words(stemming_map: Dict[str, str]) -> Dict[str, str]:
    """ Set haha and ahah variations to a simple `haha`"""
    all_words = set(stemming_map.keys())
    regex = re.compile(r"^(ha)\1+$|^(ah)\2+a?$")
    laugh_words = list(filter(regex.search, list(all_words)))

    for laugh_word in laugh_words:
        stemming_map[laugh_word] = 'laugh'
    return stemming_map
