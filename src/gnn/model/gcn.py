import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(
        self, num_classes: int, num_input_features: int, num_hidden_features: int, dropout_ratio: float, use_bias: bool
    ):
        super(GCN, self).__init__()
        self.gc_1 = GraphConvLayer(
            in_features=num_input_features, out_filters=num_hidden_features, dropout_ratio=dropout_ratio, bias=use_bias
        )
        self.gc_2 = GraphConvLayer(
            in_features=num_hidden_features, out_filters=num_classes, dropout_ratio=dropout_ratio, bias=use_bias
        )

    def forward(self, x: torch.FloatTensor, adjacency: torch.sparse.FloatTensor) -> torch.FloatTensor:
        z = F.relu(self.gc_1(x, adjacency))
        z = self.gc_2(z, adjacency)
        return z


class GraphConvLayer(nn.Module):
    """ A Graph Convolution Layer as per https://arxiv.org/pdf/1609.02907.pdf with Glorot initialisation """

    def __init__(self, in_features: int, out_filters: int, dropout_ratio: float, bias: bool):
        """
        Weight matrix / filter parameters: (in_features, out_filters) or (in_features, out_features)
        """
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_filters = out_filters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_filters))
        self.dropout = nn.Dropout(p=dropout_ratio)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_filters))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Reset all trainable parameters using Glorot & Bengio (2010) initialisation """
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_x: torch.FloatTensor, adjacency: torch.sparse.FloatTensor) -> torch.FloatTensor:
        """
        input_x.shape == (num_nodes, in_features)
        support.shape == (num_nodes, out_features)
        adjacency.shape == (num_nodes, num_nodes)
        output_z.shape == (num_nodes, out_features)
        """
        input_x = self.dropout(input_x)
        support = torch.mm(input_x, self.weight)
        output_z = torch.spmm(adjacency, support)

        if self.bias is not None:
            output_z = output_z + self.bias
        return output_z
