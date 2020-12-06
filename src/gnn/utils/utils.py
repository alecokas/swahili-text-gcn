import os
import torch

from utils.utils import rm_file


def get_device(use_gpu: bool) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def remove_previous_best(directory: str, current_step: int) -> None:
    for item in os.listdir(directory):
        if f'model-{current_step}' not in item:
            rm_file(os.path.join(directory, item))
