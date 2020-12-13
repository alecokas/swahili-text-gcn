from shared.utils import read_jsonl, save_dict_to_json


def create_stemming_map(raw_path_name: str, cleaned_path_name: str):
    raw_salama_results = read_jsonl(raw_path_name)
    stemming_map = {
        raw_result['word']: raw_result['stem'] if raw_result['stem'] != '' else raw_result['word']
        for raw_result in raw_salama_results
    }
    save_dict_to_json(cleaned_path_name, stemming_map)
