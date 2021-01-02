import torch
import torch.nn as nn
from torch.nn import functional as F


class SpGAT(nn.Module):
    """ Sparse GAT model based on https://github.com/Diego999/pyGAT/ """
    def __init__(
        self,
        num_classes: int,
        num_input_features: int,
        num_hidden_features_per_head: int,
        num_heads: int,
        dropout_ratio: float,
        relu_negative_slope: float,
    ):
        super(SpGAT, self).__init__()
        self.gat_layer_1 = [
            SpGraphAttentionLayer(
                in_features=num_input_features,
                out_features=num_hidden_features_per_head,
                dropout=dropout_ratio,
                relu_negative_slope=relu_negative_slope,
            )
            for _ in range(num_heads)
        ]
        for idx, head in enumerate(self.gat_layer_1):
            self.add_module('gat_layer_1_head_{}'.format(idx), head)

        self.out_att = SpGraphAttentionLayer(
            in_features=num_hidden_features_per_head * num_heads,
            out_features=num_classes,
            dropout=dropout_ratio,
            relu_negative_slope=relu_negative_slope,
        )
        self.dropout_ratio = dropout_ratio

    def forward(self, x: torch.sparse.FloatTensor, adj: torch.sparse.FloatTensor):
        x = F.dropout(x, self.dropout_ratio, training=self.training)
        x = torch.cat([F.elu(att(x, adj)) for att in self.gat_layer_1], dim=1)
        x = F.dropout(x, self.dropout_ratio, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x

class SpecialSpmmFunction(torch.autograd.Function):
    """ Special function for only sparse region backpropataion layer. """

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False, 'indices.requires_grad is True. Require False.'
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features: int, out_features: int, dropout: float, relu_negative_slope: float):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.relu_negative_slope = relu_negative_slope

        self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.alpha = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.relu_negative_slope)
        self.special_spmm = SpecialSpmm()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.alpha.data, gain=1.414)

    def forward(self, h: torch.sparse.FloatTensor, adj: torch.sparse.FloatTensor):
        """
        Forward pass the sprase GAT layer.
        Inputs:
            h.shape == (num_nodes, in_features)
            adjacency.shape == (num_nodes, num_nodes)
        Outputs:
            out.shape == out_features
        """
        num_nodes = h.size()[0]

        # h over here is actually w_h_transpose: we use the transposed form - in the end this doesn't matter
        # w_h_transpose.shape == (N, out_features)
        h = torch.mm(h, self.weight)
        assert not torch.isnan(h).any()

        # (Z, E) tensor of indices where there are Z non-zero indices
        edge = adj.nonzero().t()
        # edge_h.shape == (2*D x E) - Select relevant Wh nodes
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()

        """ Self-attention on the nodes - Shared attention mechanism """
        # edge_e.shape == (E) - This is the numerator in equation 3
        edge_e = torch.exp(self.leakyrelu(self.alpha.mm(edge_h).squeeze()))  # I just removed the minus here
        assert not torch.isnan(edge_e).any()
        # e_rowsum.shape == (num_nodes x 1) - This is the denominator in equation 3
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([num_nodes, num_nodes]), torch.ones(size=(num_nodes, 1)))

        # h_prime.shape == (num_nodes x out)
        # Aggregate Wh using the attention coefficients
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([num_nodes, num_nodes]), h)
        assert not torch.isnan(h_prime).any()

        # h_prime.shape == (num_nodes x out)
        # Divide by normalising factor
        h_prime = h_prime.div(e_rowsum)

        assert not torch.isnan(h_prime).any()
        return h_prime
