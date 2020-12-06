import argparse
import os
import sys

from doc2vec.train import read_and_format_docs, separate_into_subsets, train
from utils.global_constants import RES_DIR


def parse_arguments(args_to_parse):
    """ Parse CLI arguments """
    descr = 'Train Doc2Vec representation of the corpus'
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
        '--doc2vec-dir',
        type=str,
        default='doc2vec',
        help="The name of the subdirectory where we should save training data (losses, metrics, models, etc.)",
    )

    training = parser.add_argument_group('Training settings')
    training.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="The number of epochs to run",
    )
    training.add_argument(
        '--feature-dims',
        type=int,
        default=300,
        help="The number of feature dimensions to use to generate the document vectors",
    )
    return parser.parse_args(args_to_parse)


def main(args):
    """ Entry point for training a doc2vec model """
    preproc_dir = os.path.join(RES_DIR, args.input_data_dir)
    results_dir = os.path.join(RES_DIR, args.name)

    tagged_docs = read_and_format_docs(df_path=os.path.join(preproc_dir, 'dataset.csv'), text_column='document_content')
    train_docs, val_docs = separate_into_subsets(tagged_docs, preproc_dir)

    print('Training doc2vec ...')
    model = train(docs=train_docs, feature_dims=args.feature_dims, num_epochs=args.epochs)

    print('Saving doc2vec model...')
    model.save(os.path.join(results_dir, 'doc2vec-model.sav'))

    # TODO: Assess & Test the model


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
