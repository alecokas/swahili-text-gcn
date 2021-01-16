import argparse
import logging
import numpy as np
import os
import random
from sklearn.linear_model import LogisticRegression
import sys

from baselines.vectorizers import build_vectorizer_from_df, load_vectorized_data
from baselines.avg_fasttext import build_avg_fasttext_from_df, load_avg_fasttext
from baselines.doc2vec import build_doc2vec_from_df, load_doc2vec
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
    general.add_argument(
        '--model',
        type=str,
        default='tf-idf',
        choices=['tf-idf', 'count', 'doc2vec', 'fasttext'],
        help='Select the model type to use before feeding into a logistic regression layer',
    )
    general.add_argument("--seed", type=int, default=12321, help='Random seed for reproducability')

    training = parser.add_argument_group('Training settings')
    training.add_argument(
        '--train-set-label-proportion',
        type=float,
        default=0.2,
        help='Ratio of nodes in the training set which we keep labelled',
    )
    # CLI options of the form `--doc2vec-XXXX` pertain to doc2vec
    training.add_argument(
        '--doc2vec-epochs',
        type=int,
        default=10,
        help="The number of epochs to run when training Doc2Vec",
    )
    training.add_argument(
        '--doc2vec-feature-dims',
        type=int,
        default=300,
        help="The Doc2vec feature vector size",
    )
    training.add_argument(
        '--doc2vec-dm',
        type=int,
        choices=[0, 1],
        default=1,
        help="The training regime to use for Doc2Vec: Distributed Memory (1) or Distributed Bag of Words (0)",
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

    if args.model == 'tf-idf' or args.model == 'count':
        if not os.path.isdir(preproc_dir):
            os.makedirs(preproc_dir, exist_ok=True)
            build_vectorizer_from_df(
                vectorizer_name=args.model,
                save_dir=preproc_dir,
                df_path=os.path.join(RES_DIR, args.input_data_dir, 'dataset.csv'),
                stemming_map_path=os.path.join(RES_DIR, args.stemmer_path),
                text_column='document_content',
                label_column='document_type',
            )

        print(f'Load {args.model} data...')
        input_features, labels = load_vectorized_data(preproc_dir, args.model)

    elif args.model == 'fasttext':
        if not os.path.isdir(preproc_dir):
            os.makedirs(preproc_dir, exist_ok=True)
            build_avg_fasttext_from_df(
                save_dir=preproc_dir,
                df_path=os.path.join(RES_DIR, args.input_data_dir, 'dataset.csv'),
                stemming_map_path=os.path.join(RES_DIR, args.stemmer_path),
                text_column='document_content',
                label_column='document_type',
            )

        print('Load average FastText data...')
        input_features, labels = load_avg_fasttext(preproc_dir)

    elif args.model == 'doc2vec':
        if not os.path.isdir(preproc_dir):
            os.makedirs(preproc_dir, exist_ok=True)
            build_doc2vec_from_df(
                save_dir=preproc_dir,
                df_path=os.path.join(RES_DIR, args.input_data_dir, 'dataset.csv'),
                stemming_map_path=os.path.join(RES_DIR, args.stemmer_path),
                text_column='document_content',
                label_column='document_type',
                training_regime=args.doc2vec_dm,
                embedding_dimension=args.doc2vec_feature_dims,
                num_epochs=args.doc2vec_epochs,
            )

        print('Load Doc2vec data...')
        input_features, labels = load_doc2vec(preproc_dir)

    else:
        raise Exception(f'Unrecognised model type: {args.model}')

    train_nodes, val_nodes = load_train_val_nodes(
        preproc_dir=os.path.join(RES_DIR, args.input_data_dir),
        train_set_label_proportion=args.train_set_label_proportion,
        as_numpy=True,
    )

    print('Train classifier ...')
    classifier = LogisticRegression(random_state=1).fit(input_features[train_nodes, :], labels[train_nodes])
    print('Get accuracies...')
    train_predictions = classifier.predict(input_features[train_nodes, :])
    val_predictions = classifier.predict(input_features[val_nodes, :])

    train_accuracy = sum(train_predictions == labels[train_nodes]) / len(train_predictions)
    val_accuracy = sum(val_predictions == labels[val_nodes]) / len(val_predictions)

    print(f'Train Accuracy: {train_accuracy}')
    print(f'Validation Accuracy: {val_accuracy}')

    output_save_dir = os.path.join(results_dir, 'model')
    os.makedirs(output_save_dir, exist_ok=True)
    save_dict_to_json(
        {'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy}, os.path.join(output_save_dir, 'metric.json')
    )

    # from sklearn.model_selection import learning_curve
    # train_sizes, train_scores, test_scores = learning_curve(
    #     classifier, input_features[train_nodes, :], labels[train_nodes]
    # )
    # print(train_scores)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
