## Comparing traditional NLP techniques to Graph Convolutional Network for Swahili News Classification
This work empirically demonstrates the ability of Text Graph Convolutional Network (Text GCN) to outperform traditional natural language processing benchmarks for the task of semi-supervised Swahili news classification. In particular, we focus our experimentation on the sparsely-labelled semi-supervised context which is representative of the practical constraints facing low-resourced African languages. We follow up on this result by introducing a variant of the Text GCN model which utilises a bag of words embedding rather than a naive one-hot encoding to reduce the memory footprint of Text GCN whilst demonstrating similar predictive performance.

This repo contains the code used to generate the experimental results presented in the following two papers:
-  Alexandros Kastanos and Tyler Martin, [**Graph Convolutional Network for Swahili News Classification**](https://arxiv.org/abs/2103.09325), European Chapter of the Association for Computational Linguistics, AfricaNLP, 2021
-  Alexandros Kastanos and Tyler Martin, [**Graph Convolutional Network for Swahili News Classification**](https://sigtyp.io/workshops/2021/abstracts/13.pdf), North American Chapter of the Association for Computational Linguistics, SIGTYP, 2021

## Prerequisites
This project is compatible with Python >= 3.6.9. Install the full list of dependencies in your virtual environemnt by running:
```code
pip install -r requirements.txt
```
NOTE: This repo was developed (and will continue to be maintained) exclusively for Linux and Mac. The exact package versions are valid for Python 3.6.9.

Before you get started, open a Python prompt from your virtual environment and run the following to download the `punkt` data:
```Python
>>> import nltk
>>> nltk.download('punkt')
```

## Download the data
The raw dataset can be downloaded, preprocessed, and organised into a DataFrame by running the following command:
```code
python src/create_dataset.py <DOWNLOADED_DATA_DIR_NAME> --dataset-name <DATASET_NAME>
```
This script is set up to handle `Helsinki Corpus of Swahili 2.0 Not Annotated Version` and the Zenodo `Swahili : News Classification Dataset`. You select which dataset to process using the `--dataset-name` CLI option. The dataset split is generated by this script as well.

It is worth noting that for the purposes of replicating our work in the paper listed above, you only need to consider the Swahili news classification dataset.

## Preprocessing
We can apply preprocessing (stemming, error removal, etc.) using the script below:
```code
python src/generate_stemming_map.py \
    --results-dir <PREPROC_DATA_DIR_NAME> \
    --input-data-dir  <DOWNLOADED_DATA_DIR_NAME>
```
This gives us our cleaned stemming map and vocabulary counts as the main outputs.

## Train baseline models
A number of baseline models are set up to be trained out-the-box. They all use a feature generation stage followed by a logistic regression classifier. Here's an excerpt from our paper where a set of baseline models are compared to two Text GCN variants. For more details, see Table 2 in our paper.

<img src="res/table-comparison.png" width="400">

To train any of these baseline models, simply use the following:
```code
python src/train_baseline.py <BASELINE_RESULTS_DIR> \
    --input-data-dir <DOWNLOADED_DATA_DIR_NAME> \
    --stemmer-path <PREPROC_DATA_DIR_NAME>/stemming/stemming_cleaned.json \
    --model <BASELINE_MODEL>
```

## Train Text GCN models
There are two Text GCN variant models which can be trained: the vanilla `Text GCN` and `Text GCN-t2v`. You can refer to the figure below for a comparison of these models to selected baseline models as the proportion of training labels is varied. See figure 3 in our paper for full details.

<img src="res/varying-label-proportion.png" width="400">


To train a GCN model, run:
```code
python src/train_gnn.py <GNN_RESULTS_DIR> \
    --input-data-dir <DOWNLOADED_DATA_DIR_NAME> \
    --stemmer-path <PREPROC_DATA_DIR_NAME>/stemming/stemming_cleaned.json \
    --model <GNN_MODEL> \
    --input-features <NODE_REREPRESENTATION> \
    ....
```
As with all the above scripts, you can use the `--help` flag to see the full set of CLI options.

## Cite our work
```
@misc{kastanos2021graph,
      title={Graph Convolutional Network for Swahili News Classification}, 
      author={Alexandros Kastanos and Tyler Martin},
      year={2021},
      eprint={2103.09325},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Cite the datasets

Zenodo's Swahili News Dataset:
```
@misc{davis_david_2020_4300294,
  author       = {Davis David},
  title        = {Swahili : News Classification Dataset},
  month        = dec,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.4300294},
  url          = {https://doi.org/10.5281/zenodo.4300294}
}
```

Helsinki Corpus of Swahili 2.0:
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
