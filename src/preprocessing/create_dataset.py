import argparse
import os
import pandas as pd
import sys
from typing import Dict, List
import urllib.request
import zipfile

from utils import mkdir


RES_DIR = 'results'
DATA_DIR = 'data'
ROOT_DOWNLOAD_URL = 'https://korp.csc.fi/download/HCS/na-v2'
SUBDIR = 'hcs-na-v2'


def download_raw_data(download_location: str) -> None:
    """
    Download and extract the Helsinki Swahili Corpus. There are two relevant files to download:
    https://korp.csc.fi/download/HCS/na-v2/hcs-na-v2.zip
    https://korp.csc.fi/download/HCS/na-v2/hcs-na-v2.zip.md5
    """
    if os.path.isdir(download_location):
        print(f'Skipping download: {download_location} already exists')
    else:
        mkdir(download_location)
        for file_name in [f'{SUBDIR}.zip', f'{SUBDIR}.zip.md5']:
            file_url = f'{ROOT_DOWNLOAD_URL}/{file_name}'
            urllib.request.urlretrieve(file_url, f'{download_location}/{file_name}')

        with zipfile.ZipFile(os.path.join(download_location, f'{SUBDIR}.zip'), "r") as zipfd:
            zipfd.extractall(download_location)


def read_and_format_as_df(results_dir: str, data_dir: str) -> pd.DataFrame:
    """ """
    _ = os.path.join(data_dir, SUBDIR)


def generate_labels(
    results_dir: str,
    data_dir: str,
    labels_path: str,
    all_tags: List[str],
    product_df: pd.DataFrame,
    tag_struct_df: pd.DataFrame,
) -> Dict[str, str]:
    pass


def parse_arguments(args_to_parse):
    """ Parse CLI arguments """
    descr = 'Apply pre-processing to generate vegan/veggie/meat dataset'
    parser = argparse.ArgumentParser(description=descr)

    general = parser.add_argument_group('General settings')
    general.add_argument('name', type=str, help="The name of the results directory - used for saving and loading.")
    general.add_argument(
        '--output-dataset', type=str, default='dataset.csv', help="The name of the final processed dataset"
    )
    general.add_argument(
        '--output-json-labels',
        type=str,
        default='labels.json',
        help="JSON file inwhich to save the label name to index mapping",
    )
    return parser.parse_args(args_to_parse)


def main(args):
    """ Primary entry point for data pre-processing """
    data_dir = os.path.join(DATA_DIR, 'helsinki-swahili-corpus-v2-unannotated')
    results_dir = os.path.join(RES_DIR, args.name)
    mkdir(results_dir)

    download_raw_data(download_location=data_dir)

    # Get the dataset and labels into memory
    # labels_path = os.path.join(results_dir, args.output_json_labels)
    dataframe_path = os.path.join(results_dir, args.output_dataset)
    if os.path.isfile(dataframe_path):
        print(f'Skipping preprocessing, {dataframe_path} already exists')
    else:
        dataset_df = read_and_format_as_df(results_dir, data_dir)
        # save_dict_to_json(labels, labels_path)
        dataset_df.to_csv(dataframe_path, index=False, sep=';')


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
