import numpy as np
import os
from scipy.sparse import csr_matrix, diags, load_npz
import torch
from typing import Tuple


def load_datasets(graph_dir: str) -> Tuple[torch.sparse.FloatTensor, torch.FloatTensor, torch.LongTensor]:
    labels = torch.load(os.path.join(graph_dir, 'labels.pt'))
    input_features = torch.load(os.path.join(graph_dir, 'input_features.pt')).to_dense()
    adjacency = load_npz(os.path.join(graph_dir, 'adjacency.npz'))

    norm_adjacency = _normalise_adjacency(adjacency).tocoo()
    norm_adjacency = torch.sparse.FloatTensor(
        torch.LongTensor([norm_adjacency.row.tolist(), norm_adjacency.col.tolist()]),
        torch.FloatTensor(norm_adjacency.data),
    )

    return norm_adjacency, input_features, labels


def _normalise_adjacency(adjacency: csr_matrix) -> csr_matrix:
    """ A_tidle = D^(-0.5) A D^(-0.5) """
    # Create D^(-0.5)
    degree_inv_sqrt = np.power(np.array(adjacency.sum(1)), -0.5).flatten()
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    degree_inv_sqrt = diags(degree_inv_sqrt, format="coo")
    # Compute D^(-0.5) A D^(-0.5)
    return degree_inv_sqrt.dot(adjacency).dot(degree_inv_sqrt)
