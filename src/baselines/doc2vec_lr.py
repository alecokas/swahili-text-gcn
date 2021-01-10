from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from sklearn.feature_extraction.text import CountVectorizer
from typing import List

from shared.utils import read_json_as_dict, tokenize_prune_stem, write_to_meta
from shared.loaders import load_text_and_labels, save_categorical_labels


def build_doc2vec_from_df(
    save_dir: str,
    df_path: str,
    stemming_map_path: str,
    text_column: str,
    label_column: str,
    training_regime: int,
    embedding_dimension: int,
    num_epochs: int,
) -> None:
    if not os.path.isfile(df_path):
        raise FileNotFoundError(
            f'{df_path} could not be found.\
                Remember that you first need to generate the dataset using the `create_dataset` script'
        )
    if not os.path.isfile(stemming_map_path):
        raise FileNotFoundError(
            f'{stemming_map_path} could not be found.\
                Remember that you need to first generate a stemming map using the `download_stemming` script'
        )
    stemming_map = read_json_as_dict(stemming_map_path)
    document_list, labels = load_text_and_labels(df_path, text_column, label_column)
    save_categorical_labels(save_dir, labels, as_numpy=True)

    # Tokenize
    cv = CountVectorizer(tokenizer=lambda text: tokenize_prune_stem(text, stemming_map=stemming_map))
    cv_tokenizer = cv.build_tokenizer()
    document_list = [cv_tokenizer(document) for document in document_list]

    # Convert to TaggedDocument and train
    document_list = [TaggedDocument(doc, [i]) for i, doc in enumerate(document_list)]
    doc2vec_model = _train_doc2vec(
        docs=document_list,
        feature_dims=embedding_dimension,
        num_epochs=num_epochs,
        training_regime=training_regime,
    )

    doc2vec_map = doc2vec_model.dv
    print(f'There are {len(doc2vec_map)} documents in our corpus')
    # TODO: Save it

    word_vector_map = doc2vec_model.wv
    print(f'Our vocabulary has {len(word_vector_map)} words in it')
    print(word_vector_map)

    # Save and meta-data to disk
    write_to_meta(
        data_meta_path=os.path.join(save_dir, 'meta.json'),
        key_val={
            'training_regime': 'PV-DM' if training_regime == 1 else 'PV-DBOW',
            'num_docs': len(document_list),
            'vector_size': embedding_dimension,
        },
    )


def _train_doc2vec(docs: List[TaggedDocument], feature_dims: int, num_epochs: int, training_regime: int) -> Doc2Vec:
    model = Doc2Vec(vector_size=feature_dims, window=2, min_count=2, workers=4, epochs=num_epochs, dm=training_regime)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=num_epochs)
    return model
