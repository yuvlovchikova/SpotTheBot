"""

Lemmatization

"""

import math
import glob
import stanza
import numpy as np
import pandas as pd
from tqdm import tqdm

stanza.download('en')

#It turns out that we have about 13000 documents containing at least 1000 words. Let's take them.
THRESHOLD = 1000
START_INDEX = 0
GRADUAL_NUMBER = 10
CHECK_POINT_SHAPE = 1000
PATH = "EnGroupedTexts.csv"


def select(dataframe: pd.DataFrame, threshold: int) -> pd.DataFrame:
    lambda_ = lambda text: len(text.split()) > threshold
    corpus = dataframe[dataframe['proc_sentence'].apply(lambda_)].reset_index(drop=True)
    return corpus

def _lemmatize_string(string: str) -> str:
    result = []
    nlp = stanza.Pipeline('en', processors='tokenize, lemma, pos', logging_level='FATAL')
    doc = nlp(string)
    for word in doc.iter_words():
        result.append(word.lemma)
    return ' '.join(result)

def _lemmatize_sentences(sentences: np.ndarray) -> list:
    in_sentences = [stanza.Document([], text=sentence) for sentence in sentences]
    nlp = stanza.Pipeline('en', processors='tokenize, lemma, pos', logging_level='FATAL')
    out_sentences = nlp(in_sentences)
    result_sentences = []
    for sentence in out_sentences:
        new_sentence = []
        for word in sentence.iter_words():
            new_sentence.append(word.lemma)
        result_sentences.append(' '.join(new_sentence))
    return result_sentences

def gradualLemma(corpus: pd.DataFrame,
                 gradual_number: int,
                 check_point_shape: int,
                 start_index: int) -> pd.DataFrame:
    corpus = corpus[start_index:].reset_index(drop=True)
    lemmatized_corpus = pd.DataFrame(columns=['pg_index', 'lem_sentence'])
    lemmatized_corpus_ = pd.DataFrame(columns=['pg_index', 'lem_sentence'])
    n_iters = math.ceil(corpus.shape[0] / gradual_number)
    data_count = 1
    for i in tqdm(range(n_iters)):
        if corpus.shape[0] == 0:
            break
        slice_ = corpus[:gradual_number]
        lemmatized_slice = _lemmatize_sentences(slice_['proc_sentence'].values)
        lemmatized_corpus_ = lemmatized_corpus_.append(pd.DataFrame({'pg_index': slice_['pg_index'], 
                                                                     'lem_sentence': lemmatized_slice}))
        corpus = corpus[gradual_number:]
        if lemmatized_corpus_.shape[0] >= check_point_shape or i == n_iters - 1:
            lemmatized_corpus = lemmatized_corpus.append(lemmatized_corpus_)
            lemmatized_corpus_ = pd.DataFrame(columns=['pg_index', 'lem_sentence'])
            name_ = "EnLemCorpus/EnLemCorpus" + str(data_count) + ".csv"
            lemmatized_corpus.to_csv(name_, index=False)
            print(name_, "saved")
            data_count += 1
    return lemmatized_corpus

def lemmatize_dataframe(path: str,
                        threshold: int,
                        gradual_number: int,
                        check_point_shape: int,
                        start_index: int) -> pd.DataFrame:
    dataframe = pd.read_csv(path).dropna()
    selected_dataframe = select(dataframe, threshold)
    lemmatized_dataframe = gradualLemma(selected_dataframe,
                                        gradual_number,
                                        check_point_shape,
                                        start_index)
    return lemmatized_dataframe

def concatinate(folder_path: str) -> pd.DataFrame:
    dataframe = pd.DataFrame()
    files = []
    for filename in glob.iglob(folder_path + "/*", recursive=True):
        files.append(filename)
    for file_path in files:
        current_dataframe = pd.read_csv(file_path)
        dataframe = pd.concat([dataframe, current_dataframe])
    return dataframe


lemmatized_corpus = lemmatize_dataframe(PATH, THRESHOLD, GRADUAL_NUMBER,
                                        CHECK_POINT_SHAPE, START_INDEX)
lemmatized_dataframe = concatinate("EnLemCorpus").sort_values(by='pg_index').reset_index(drop=True)
lemmatized_dataframe.to_csv("EnLemCorpus.csv", index=False)