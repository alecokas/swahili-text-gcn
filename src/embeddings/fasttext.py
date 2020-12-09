import fasttext.util
import numpy as np
import os
import shutil
from typing import List


def load_pretrained_swahili_fasttext(save_location: str):
    """
    Model was trained using CBOW with position-weights, in dimension 300,
    with character n-grams of length 5,a window of size 5 and 10 negatives
    """
    model_name = 'cc.sw.300.bin'
    target_path_name = os.path.join(save_location, model_name)

    if not os.path.isfile(target_path_name):
        fasttext.util.download_model('sw', if_exists='ignore')
        shutil.move(model_name, target_path_name)
    return fasttext.load_model(target_path_name)


def generate_embeddings(ft_model, word_list: List[str]) -> List[np.ndarray]:
    return [ft_model.get_word_vector(word) for word in word_list]
