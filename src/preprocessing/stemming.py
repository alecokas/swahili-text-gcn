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


def _generate_initial_map(raw_salama_results: List[Dict[str, str]]) -> Dict[str, str]:
    return {
        raw_result['word']: raw_result['stem'] if raw_result['stem'] != '' else raw_result['word']
        for raw_result in raw_salama_results
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


def _merge_repeated_letter_entries(stemming_map: Dict[str, str]) -> Dict[str, str]:
    """
    XXX: For example merge the pairs: {`aisee`: `aisee`} and {`aiseee`: `aiseee`}
    NOTE: Not actually convinced that this is a good idea as it might merge things that shouldn't be merged
    """
    all_words = set(stemming_map.keys())
    regex = re.compile(r"^([a-z])\1\1+|([a-z])\2\2+$")
    words_with_repeated_chars = list(filter(regex.search, list(all_words)))

    print(words_with_repeated_chars)
    # TODO: roll back the repeated character
    for word_with_repeated_chars in words_with_repeated_chars:
        match = regex.search(word_with_repeated_chars)
        print(match.span(0))
        left_idx, right_idx = match.span(0)
        print(word_with_repeated_chars[left_idx:right_idx])
        print(word_with_repeated_chars)
        if left_idx == 0:
            # Starting repeated character
            while left_idx < right_idx:
                left_idx += 1
                if word_with_repeated_chars[left_idx:] in all_words:
                    # Merge stemmed form to match the non-repeating form
                    print(word_with_repeated_chars[left_idx:])
                    stemming_map[word_with_repeated_chars] = stemming_map[
                        word_with_repeated_chars[left_idx:]
                    ]
        else:
            # Trailing repeating character
            while left_idx < right_idx:
                right_idx -= 1
                if word_with_repeated_chars[:right_idx] in all_words:
                    print(word_with_repeated_chars[:right_idx])
                    stemming_map[word_with_repeated_chars] = stemming_map[
                        word_with_repeated_chars[:right_idx]
                    ]
        print()

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
