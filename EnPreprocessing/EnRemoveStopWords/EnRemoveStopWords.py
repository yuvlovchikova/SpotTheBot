"""

Remove Stop Words

"""

import json
import numpy as np
import pandas as pd

LEMATIZATED_CORPUS = "EnLemCorpus.csv"
STOPWORDS = "stopwords-en.json"
STOPWORDS_ADD = ["a", "aa", "aaa", "aaaa"]
TEXT_SHAPES = [500, 1000, 5000]


def remove_stop_words_text(text: str, stop_words: list) -> str:
    text_list = text.split()
    result_text_list = []
    for word in text_list:
        if word not in stop_words:
            result_text_list.append(word)
    return ' '.join(result_text_list)

def remove_stop_words(lemmatized_corpus: str, stop_words: str, add_: list) -> pd.DataFrame:
    lemmatized_corpus = pd.read_csv(LEMATIZATED_CORPUS)
    stop_words = json.load(open(STOPWORDS)) + add_
    lambda_ = lambda text: remove_stop_words_text(text, stop_words)
    lemmatized_corpus['clean_text'] = lemmatized_corpus['lem_sentence'].apply(lambda_)
    return lemmatized_corpus

def create_textshape_data(dataframe: pd.DataFrame, text_shape: list, column: str) -> pd.DataFrame:
    data_text_shape = pd.DataFrame(columns=['text_shape', 'pages_amount'])
    for shape in text_shape:
        lambda_ = lambda text: len(text.split()) > shape
        pages_amount = dataframe[dataframe[column].apply(lambda_)].shape[0]
        data_text_shape = data_text_shape.append({'text_shape': shape,
                                                  'pages_amount': pages_amount}, ignore_index=True)
    return data_text_shape


clean_corpus = remove_stop_words(LEMATIZATED_CORPUS, STOPWORDS, STOPWORDS_ADD)

# create_textshape_data(clean_corpus, TEXT_SHAPES, 'clean_text')
# After removing the stop words, we got about the same number of pages

clean_corpus[['clean_text']].to_csv("EnCleanCorpus.csv", index=False)
