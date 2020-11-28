import math
import torch
import torch.nn as nn
from torch.nn import functional as F


NEG_MASK = -9e10


class GAT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_input_features: int,
        num_hidden_features_per_head: int,
        num_heads: int,
        dropout_ratio: float,
        relu_negative_slope: float,
    ):
        super(GAT, self).__init__()
        self.gat_layer_1 = [
            GraphAttnLayer(
                in_features=num_input_features,
                out_features=num_hidden_features_per_head,
                dropout_ratio=dropout_ratio,
                relu_negative_slope=relu_negative_slope,
            )
            for i in range(num_heads)
        ]
        for idx, head in enumerate(self.gat_layer_1):
            self.add_module('gat_layer_1_head_{}'.format(idx), head)

        self.gat_layer_2 = GraphAttnLayer(
            in_features=num_hidden_features_per_head * num_heads,
            out_features=num_classes,
            dropout_ratio=dropout_ratio,
            relu_negative_slope=relu_negative_slope,
        )

    def forward(self, x: torch.FloatTensor, adjacency: torch.sparse.FloatTensor):
        h = torch.cat([F.elu(gat_layer_1_head(x, adjacency)) for gat_layer_1_head in self.gat_layer_1], dim=1)
        return F.elu(self.gat_layer_2(h, adjacency))


class GraphAttnLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_ratio: float, relu_negative_slope: float):
        super(GraphAttnLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(p=dropout_ratio)

        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.alpha = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.relu = nn.LeakyReLU(relu_negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        """ Reset all trainable parameters using Glorot & Bengio (2010) initialisation """
        alpha_stdv = 1.0 / math.sqrt(self.alpha.size(1))
        weights_stdv = 1.0 / math.sqrt(self.weights.size(1))
        self.alpha.data.uniform_(-alpha_stdv, alpha_stdv)
        self.weights.data.uniform_(-weights_stdv, weights_stdv)

    def forward(self, h: torch.FloatTensor, adjacency: torch.sparse.FloatTensor):
        """
        Forward pass the GAT layer.
        Inputs:
            h.shape == (N, in_features)
            adjacency.shape == (N, N)
        Outputs:
            out.shape == out_features
        """
        # w_h_transpose: we use the transposed form - in the end this doesn't matter
        # w_h_transpose.shape == (N, out_features)
        h = self.dropout(h)
        w_h_transpose = torch.mm(h, self.weights)
        e = self._attn_mechanism(w_h_transpose)
        a = self._norm_over_neighbourhood(e, adjacency)
        return torch.matmul(a, w_h_transpose)

    def _attn_mechanism(self, w_h_transpose: torch.FloatTensor):
        """
        Attention mechanism
       Input:
            w_h_transpose.shape == (N, out_features)
        Output:
            e.shape == (num_nodes, num_nodes)
        """
        Wh_combinations = self._transform_attn_input(w_h_transpose)
        # torch.matmul(Wh_combinations, self.alpha).shape
        # -> (num_nodes, num_nodes, 2 * out_features) X (2 * out_features, 1)
        # -> (num_nodes, num_nodes, 1) -> (num_nodes, num_nodes)
        return self.relu(torch.matmul(Wh_combinations, self.alpha).squeeze(2))

    def _transform_attn_input(self, w_h_transpose: torch.FloatTensor):
        """
        Create a representation where with each concatenated Wh_{i,j} features
        Input:
            w_h_transpose.shape == (N, out_features)
        Output:
            Wh_combinations.shape == (num_nodes, num_nodes, 2 * self.out_features)
        """
        num_nodes = w_h_transpose.shape[0]
        Wh_repeated_interleave = w_h_transpose.repeat_interleave(num_nodes, dim=0)
        Wh_repeated = w_h_transpose.repeat(num_nodes, 1)
        Wh_combinations = torch.cat([Wh_repeated_interleave, Wh_repeated], dim=1)

        return Wh_combinations.view(num_nodes, num_nodes, 2 * self.out_features)

    def _norm_over_neighbourhood(self, e: torch.FloatTensor, adjacency: torch.sparse.FloatTensor):
        """
            Mask out elements in e where adjacency is <= 0 by replacing those elements with NEG_MASK
            thereby only normalising using nodes in the neighbourhood using a softmax.

            Inputs:
                e.shape == (num_nodes, num_nodes)
                adjacency.shape == (num_nodes, num_nodes)
            Output:
                norm_e.shape == (num_nodes)
        """
        mask_tensor = NEG_MASK * torch.ones_like(e)
        # NOTE: Perhaps something interesting to be done where with the threshold
        masked_e = torch.where(adjacency > 0, e, mask_tensor)
        return F.softmax(masked_e, dim=1)
