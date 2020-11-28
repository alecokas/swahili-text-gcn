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

## Running the code
This project is written in Python (>=3.6.9) and PyTorch. Install the full list of dependencies in your virtual environemnt by running:
```code
pip install -r requirements.txt
```

### Download and data pre-processing
The `Helsinki Corpus of Swahili 2.0 Not Annotated Version` can be downloaded, preprocessed, and organised into a DataFrame by running the following command:

```code
python src/preprocessing/create_dataset.py <RESULTS_DIRECTORY_NAME>
```