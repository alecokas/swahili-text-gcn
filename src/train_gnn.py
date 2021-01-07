import argparse
import numpy as np
import os
import random
import sys
import torch

from gnn.dataloading.build_graph import build_graph_from_df
from gnn.dataloading.loaders import load_datasets, load_train_val_nodes
from gnn.model.model import create_model
from gnn.model.trainer import Trainer
from gnn.utils.utils import get_device, get_vocab_size
from shared.utils import mkdir, save_cli_options

# TODO: import this from global_constants.py
RES_DIR = 'results'


def parse_arguments(args_to_parse):
    """ Parse CLI arguments """
    parser = argparse.ArgumentParser(description='Train a Text GCN model')

    general = parser.add_argument_group('General settings')
    general.add_argument('name', type=str, help="The name of the experimental directory - used for saving and loading.")
    general.add_argument(
        '--model',
        type=str,
        default='gcn',
        choices=['gcn', 'gat', 'spgat'],
        help='Select a GNN model to use'
    )
    general.add_argument(
        '--input-data-dir',
        type=str,
        required=True,
        help="The name of the directory in which we have a our processed corpora (in a DataFrame)",
    )
    general.add_argument(
        '--graph-data-dir',
        type=str,
        default='graph',
        help="The name of the directory in which we save the processed graph data - such as the adjacency and labels",
    )
    general.add_argument(
        '--train-dir',
        type=str,
        default='train',
        help="The name of the subdirectory where we should save training data (losses, metrics, models, etc.)",
    )
    general.add_argument(
        "--stemmer-path",
        type=str,
        required=True,
        help="Path to the SALAMA stemming dictionary",
    )
    general.add_argument(
        "--seed",
        type=int,
        default=12321,
        help='Random seed for reproducability'
    )

    training = parser.add_argument_group('Training settings')
    training.add_argument(
        '--lr',
        type=float,
        default=0.02,
        help="Learning rate",
    )
    training.add_argument(
        '--dropout-ratio',
        type=float,
        default=0.5,
        help="Dropout rate to be applied to all layers",
    )
    training.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="The number of epochs to run",
    )
    training.add_argument(
        '--use-gpu', action='store_true', default=False, help='Set this parameter to run on GPU (cuda)'
    )
    training.add_argument(
        '--train-set-label-proportion',
        type=float,
        default=0.2,
        help='Ratio of nodes in the training set which we keep labelled',
    )
    return parser.parse_args(args_to_parse)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results_dir = os.path.join(RES_DIR, args.name)
    mkdir(results_dir)
    save_cli_options(args, results_dir)

    graph_dir = os.path.join(results_dir, args.graph_data_dir)

    if not os.path.isdir(graph_dir):
        print('Building graph...')
        mkdir(graph_dir)
        build_graph_from_df(
            graph_dir=graph_dir,
            df_path=os.path.join(RES_DIR, args.input_data_dir, 'dataset.csv'),
            stemming_map_path=os.path.join(RES_DIR, args.stemmer_path),
            text_column='document_content',
            label_column='document_type',
            window_size=20,
        )

    print('Load and normalise...')
    adjacency, input_features, labels = load_datasets(graph_dir)

    train_nodes, val_nodes = load_train_val_nodes(
        preproc_dir=os.path.join(RES_DIR, args.input_data_dir),
        train_set_label_proportion=args.train_set_label_proportion,
    )

    # Initialise model, trainer, and train
    text_gcn_model = create_model(
        model_type=args.model,
        num_classes=len(labels.unique()),
        num_input_features=len(input_features),
        num_hidden_features=200,
        num_heads=8,
        dropout_ratio=args.dropout_ratio,
        use_bias=False,
        relu_negative_slope=0.2
    )
    trainer = Trainer(
        model=text_gcn_model,
        learning_rate=args.lr,
        device=get_device(args.use_gpu),
        train_nodes=train_nodes,
        val_nodes=val_nodes,
        vocab_size=get_vocab_size(graph_dir),
        results_dir=os.path.join(results_dir, args.train_dir),
        validate_every_n_epochs=2,
        save_after_n_epochs=0,
        checkpoint_every_n_epochs=2,
    )
    print('Training...')
    trainer(
        input_features=input_features,
        adjacency=adjacency,
        labels=labels,
        num_epochs=args.epochs,
    )

    print('Complete!')


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
