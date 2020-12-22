import os
import torch

from shared.utils import rm_file, read_json_as_dict


def get_device(use_gpu: bool) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def remove_previous_best(directory: str, current_step: int) -> None:
    for item in os.listdir(directory):
        if f'model-{current_step}' not in item:
            rm_file(os.path.join(directory, item))


def get_vocab_size(graph_dir: str) -> int:
    return len(read_json_as_dict(os.path.join(graph_dir, 'vocab_map.json')))
