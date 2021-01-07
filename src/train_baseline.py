import argparse
import os
import numpy as np
import random
import sys

from baselines.tfidf_lr import build_tfidf_from_df
from shared.global_constants import RES_DIR
from shared.utils import save_cli_options


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
    general.add_argument(
        "--seed",
        type=int,
        default=12321,
        help='Random seed for reproducability'
    )

    training = parser.add_argument_group('Training settings')
    training.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="The number of epochs to run",
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


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
