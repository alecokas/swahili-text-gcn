from bs4 import BeautifulSoup
import jsonlines
import os
import re
import requests
from typing import Dict, List, Set
from tqdm import tqdm

from preprocessing.text_stripper import ignore_non_ascii
from shared.utils import append_to_jsonl, read_json_as_dict, read_jsonl, save_dict_to_json


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


def remove_stemming_entries_below_count_threshold(cleaned_path_name: str, cleaned_vocab_path: str) -> None:
    """ Overwrites the file at cleaned_path_name """
    stemming_map = read_json_as_dict(cleaned_path_name)
    vocab_count_map = read_json_as_dict(cleaned_vocab_path)
    stemming_map = {
        raw_word: stemmed_word for raw_word, stemmed_word in stemming_map.items() if stemmed_word in vocab_count_map
    }
    save_dict_to_json(stemming_map, cleaned_path_name)


def get_all_completed_words(stemming_download_path: str) -> Set[str]:
    done_words = set()
    if os.path.exists(stemming_download_path):
        with jsonlines.open(stemming_download_path) as reader:
            for obj in reader:
                done_words.add(obj["word"])
    return done_words


def get_new_words_to_add(vocab_counts: dict, done_words: set, number_to_add: int, count_threshold: int) -> List[str]:
    words_above_threshold = [word for word, count in vocab_counts.items() if count >= count_threshold]
    words_to_add = [word for word in words_above_threshold if word not in done_words][:number_to_add]
    return words_above_threshold, words_to_add


def add_words_to_map(words_to_add: List[str], stemming_download_path: str) -> None:
    if len(words_to_add) == 0:
        print("All stemming data downloaded")
        return
    for word in tqdm(words_to_add):
        _query_word(stemming_download_path, word)


def _extract_stem(text: str) -> str:
    return text.split("[")[1].split("]")[0]


def _query_word(stemming_download_path: str, word: str) -> None:
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
            query_data["stem"] = _extract_stem(box.text)
        query_data["status"] = 1
    except Exception as exception:
        query_data["exception"] = str(exception)
        query_data["status"] = 0
        print(f"Exception of type {str(exception)} for word {word}")
    append_to_jsonl(stemming_download_path, query_data)
