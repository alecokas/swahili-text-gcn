from gnn.model.gcn import GCN
from gnn.model.gat import GAT
from gnn.model.sparse_gat import SpGAT


def create_model(model_type: str, **kwargs):
    if model_type == 'gcn':
        return GCN(
            num_classes=kwargs.get('num_classes'),
            num_input_features=kwargs.get('num_input_features'),
            num_hidden_features=kwargs.get('num_hidden_features'),
            dropout_ratio=kwargs.get('dropout_ratio'),
            use_bias=kwargs.get('use_bias'),
        )
    elif model_type == 'gat':
        return GAT(
            num_classes=kwargs.get('num_classes'),
            num_input_features=kwargs.get('num_input_features'),
            num_hidden_features_per_head=kwargs.get('num_hidden_features'),
            num_heads=kwargs.get('num_heads'),
            dropout_ratio=kwargs.get('dropout_ratio'),
            relu_negative_slope=kwargs.get('relu_negative_slope'),
        )
    elif model_type == 'spgat':
        return SpGAT(
            num_classes=kwargs.get('num_classes'),
            num_input_features=kwargs.get('num_input_features'),
            num_hidden_features_per_head=kwargs.get('num_hidden_features'),
            num_heads=kwargs.get('num_heads'),
            dropout_ratio=kwargs.get('dropout_ratio'),
            relu_negative_slope=kwargs.get('relu_negative_slope'),
        )
    else:
        raise NotImplementedError(f'The model type {model_type} has not yet been implemented')
