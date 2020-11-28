import errno
import json
import os
from typing import Any, Dict, List


def mkdir(directory: str) -> None:
    """ Make directory if it does not exist """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_list_from_file(file_path_name: str) -> List[str]:
    with open(file_path_name, 'r') as f:
        return [json.loads(line.strip()) for line in f.readlines()]


def load_json_to_dict(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as fp:
        return json.load(fp)


def save_dict_to_json(dict_map: Dict[str, int], file_path: str):
    with open(file_path, 'w') as fp:
        json.dump(dict_map, fp, sort_keys=True, indent=4)
