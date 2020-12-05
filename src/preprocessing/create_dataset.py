import argparse
import os
import pandas as pd
from pathlib import Path
import sys
import urllib.request
import zipfile

from text_stripper import strip_tags, ignore_non_ascii
from utils import mkdir, save_dict_to_json


RES_DIR = 'results'
DATA_DIR = 'data'
ROOT_DOWNLOAD_URL = 'https://korp.csc.fi/download/HCS/na-v2'
SUBDIR = 'hcs-na-v2'


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
    """ Read the raw data and reformat it into a DataFrame with labels """
    collated_dict = {'path': [], 'document_content': [], 'document_type': []}
    data_root = os.path.join(data_dir, SUBDIR)
    for path in Path(data_root).rglob('*.shu'):
        doc_contents = ignore_non_ascii(strip_tags(path.read_text()).lower()).strip()
        abs_path = str(path.resolve())
        document_label = _get_document_type(abs_path)
        collated_dict['path'].append(abs_path)
        collated_dict['document_content'].append(doc_contents)
        collated_dict['document_type'].append(document_label)
    return pd.DataFrame(collated_dict)


def _get_document_type(abs_path: str) -> str:
    """
    The data is downloaded such that the lowest subdirectory is the name of the document type.
    i.e.) In the path /hcs-na-v2/new-mat/bunge/han53-2005.shu, the document type is "bunge"
    """
    return abs_path.split('/')[-2]


def main(args):
    """ Primary entry point for data pre-processing """
    data_dir = os.path.join(DATA_DIR, 'helsinki-swahili-corpus-v2-unannotated')
    results_dir = os.path.join(RES_DIR, args.name)
    mkdir(results_dir)

    download_raw_data(download_location=data_dir)

    # Get the dataset and labels into a DataFrame and json file respectively
    labels_path = os.path.join(results_dir, args.output_json_labels)
    dataframe_path = os.path.join(results_dir, args.output_dataset)
    if os.path.isfile(dataframe_path):
        print(f'Skipping preprocessing, {dataframe_path} already exists')
    else:
        dataset_df = read_and_format_as_df(results_dir, data_dir)
        labels_dict = {doc_type: idx for idx, doc_type in enumerate(dataset_df['document_type'].unique().tolist())}
        save_dict_to_json(labels_dict, labels_path)
        dataset_df.to_csv(dataframe_path, index=False, sep=';')


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
