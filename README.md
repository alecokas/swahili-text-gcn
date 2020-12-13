# text-gnn

Dataset citation:
```
@misc{hcs-na-v2_en,
 author={Arvi Hurskainen and {Department of World Cultures, University of Helsinki}},
 year={2016},
 title={{Helsinki Corpus of Swahili 2.0 Not Annotated Version}},
 publisher={Kielipankki},
 type={text corpus},
 url={http://urn.fi/urn:nbn:fi:lb-2016011302},
}
```

## Prerequisites
This project is written in Python 3.6.9. Install the full list of dependencies in your virtual environemnt by running:
```code
pip install -r requirements.txt
```

## Download and general data pre-processing
The raw dataset can be downloaded, preprocessed, and organised into a DataFrame by running the following command:

```code
python src/create_dataset.py <PREPROC_RESULTS_DIRECTORY_NAME>
```
This script is set up to handle the `Helsinki Corpus of Swahili 2.0 Not Annotated Version` and the Zenodo `Swahili : News Classification Dataset`. You which dataset to process using the `--dataset-name` CLI option.

**TODO:**
```
[] - Consider excluding words which appear less than X times from the graph
```

## Generate Swahili dictionary
**TODO:**
```
[] - Stemming
python src/download_stemming.py --number-to-add 1000
```

## Training
There are a number of models which can be trained via the interface.

### Graph Neural Networks
In order to train a GNN for the document classification task, we must first build the graph representation of the corpora. This is done automatically on the first run, and thereafter the GNN model simply trains using the graph it sees in the `--grade-data-dir` directory.
```code
python src/train_gnn.py <GNN_RESULTS_DIR_NAME> --input-data-dir <PREPROC_RESULTS_DIRECTORY_NAME> --graph-data-dir graph --train-dir train
```

### Document Embeddings: Doc2vec
**TODO:**
```
[] - doc2vec as alternative to one-hot vectors for document nodes
[] - Assess & Test the model
```

### Word Embeddings: FastText
**TODO:**
```
[] - fasttext as alternative to one-hot vectors for word nodes
```