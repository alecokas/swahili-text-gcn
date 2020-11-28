import errno
import json
import os
from typing import Dict


def mkdir(directory: str) -> None:
    """ Make directory if it does not exist """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_dict_to_json(dict_map: Dict[str, int], file_path: str):
    with open(file_path, 'w') as fp:
        json.dump(dict_map, fp, sort_keys=True, indent=4)
