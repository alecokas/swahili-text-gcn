import argparse
import logging
import numpy as np
import os
import random
from sklearn.linear_model import LogisticRegression
import sys

from baselines.tfidf_lr import build_tfidf_from_df, load_tfidf
from shared.global_constants import RES_DIR
from shared.loaders import load_train_val_nodes
from shared.utils import save_cli_options, save_dict_to_json


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def parse_arguments(args_to_parse):
    """ Parse CLI arguments """
    descr = 'Train a baseline model'
    parser = argparse.ArgumentParser(description=descr)

    general = parser.add_argument_group('General settings')
    general.add_argument('name', type=str, help="The name of the experimental directory - used for saving and loading.")
    general.add_argument(
        '--input-data-dir',
        type=str,
        required=True,
        help="The name of the directory from which to load the pre-processed data",
    )
    general.add_argument(
        "--stemmer-path",
        type=str,
        required=True,
        help="Path to the SALAMA stemming dictionary",
    )
    general.add_argument("--seed", type=int, default=12321, help='Random seed for reproducability')

    training = parser.add_argument_group('Training settings')
    training.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="The number of epochs to run",
    )
    training.add_argument(
        '--train-set-label-proportion',
        type=float,
        default=0.2,
        help='Ratio of nodes in the training set which we keep labelled',
    )
    return parser.parse_args(args_to_parse)


def main(args):
    """ Entry point for training a doc2vec model """
    random.seed(args.seed)
    np.random.seed(args.seed)

    results_dir = os.path.join(RES_DIR, args.name)
    os.makedirs(results_dir, exist_ok=True)
    save_cli_options(args, results_dir)

    preproc_dir = os.path.join(results_dir, 'preproc')

    if not os.path.isdir(preproc_dir):
        os.makedirs(preproc_dir, exist_ok=True)
        build_tfidf_from_df(
            save_dir=preproc_dir,
            df_path=os.path.join(RES_DIR, args.input_data_dir, 'dataset.csv'),
            stemming_map_path=os.path.join(RES_DIR, args.stemmer_path),
            text_column='document_content',
            label_column='document_type',
        )

    print('Load data...')
    tfidf_features, labels = load_tfidf(preproc_dir)

    train_nodes, val_nodes = load_train_val_nodes(
        preproc_dir=os.path.join(RES_DIR, args.input_data_dir),
        train_set_label_proportion=args.train_set_label_proportion,
        as_numpy=True,
    )

    print('Train classifier ...')
    classifier = LogisticRegression(random_state=1).fit(tfidf_features[train_nodes, :], labels[train_nodes])
    print('Get accuracies...')
    train_predictions = classifier.predict(tfidf_features[train_nodes, :])
    val_predictions = classifier.predict(tfidf_features[val_nodes, :])

    train_accuracy = sum(train_predictions == labels[train_nodes]) / len(train_predictions)
    val_accuracy = sum(val_predictions == labels[val_nodes]) / len(val_predictions)

    print(f'Train Accuracy: {train_accuracy}')
    print(f'Validation Accuracy: {val_accuracy}')

    output_save_dir = os.path.join(results_dir, 'model')
    os.makedirs(output_save_dir, exist_ok=True)
    save_dict_to_json(
        {'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy}, os.path.join(output_save_dir, 'metric.json')
    )


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
