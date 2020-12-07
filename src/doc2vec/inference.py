from gensim.models.doc2vec import Doc2Vec
import numpy as np
from typing import List


def infer_document_embeddings(model_path: str, doc_list: List[List[str]]) -> List[np.ndarray]:
    """ NOTE: Inference is not deterministic therefore representations will vary between calls """
    model = Doc2Vec.load(model_path)
    # TODO: concat list elements sos that it is a 2D array
    return [model.infer_vector(doc) for doc in doc_list]
