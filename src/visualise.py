import argparse
import os
import sys

from gnn.dataloading.loaders import load_datasets
from baselines.vectorizers import load_vectorized_data
from baselines.avg_fasttext import load_avg_fasttext
from baselines.doc2vec import load_doc2vec
from shared.global_constants import RES_DIR
from shared.viz import tsne_plot


def parse_arguments(args_to_parse):
    """ Parse CLI arguments """
    descr = 'Visualise via t-SNE'
    parser = argparse.ArgumentParser(description=descr)

    general = parser.add_argument_group('General settings')
    general.add_argument(
        'name', type=str, help="The name of the experimental directory - used for saving visualisations."
    )
    general.add_argument(
        '--load-dir',
        type=str,
        required=True,
        help="The name of the directory from which to load the pre-processed data (excluding `results`)",
    )
    general.add_argument(
        '--data-type',
        type=str,
        default='',
        choices=['vector', 'doc2vec', 'fasttext'],
        help="The name of the data to load from",
    )
    return parser.parse_args(args_to_parse)


def main(args):
    results_dir = os.path.join(RES_DIR, args.name)
    os.makedirs(results_dir, exist_ok=True)
    load_dir = os.path.join(RES_DIR, args.load_dir)

    if args.data_type == 'vector':
        input_features, labels = load_vectorized_data(load_dir)
    elif args.data_type == 'doc2vec':
        input_features, labels = load_doc2vec(load_dir)
    elif args.data_type == 'fasttext':
        input_features, labels = load_avg_fasttext(load_dir)
    elif args.data_type == 'graph':
        _, input_features, labels = load_datasets(load_dir)
    else:
        raise ValueError(f'{args.data_type} is not a valid data_type')

    tsne_plot(input_features, labels, results_dir)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
