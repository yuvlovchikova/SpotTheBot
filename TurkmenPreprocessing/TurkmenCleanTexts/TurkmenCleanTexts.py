"""

Clean and Groupby Texts

"""

import re
import json
import nltk
import numpy as np
import pandas as pd
from devon.devon import FSMStemmer

STOPWORDS_ADD = ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaba"]

def _split_upper(word: str) -> list:
    upper_word_list = re.split("(?=[A-Z])", word)
    if upper_word_list[0] == "":
        return upper_word_list[1:]
    return upper_word_list

def _check_word_len(word: str, min_len: int = 3, max_len: int = 30) -> bool:
    if len(word) >= min_len and len(word) <= max_len:
        return word
    return None

def _delete_apostrof(word: str) -> str:
    apostrofs = ["'", "ʻ", "ʼ"]
    if word in apostrofs:
        return ""
    if word[0] in apostrofs:
        word = word[1:]
    if word[-1] in apostrofs:
        word = word[:-1]
    return word

def _check_stop_words(word: str, stop_words: list) -> str:
    if word in stop_words:
        return ""
    return word

def _clean_pipline(words_list: np.ndarray) -> pd.Series:
    words_list = pd.Series(words_list)
    words_list = words_list.apply(_delete_apostrof)
    words_list = words_list.apply(lambda word: word.lower())
    words_list = words_list.apply(lambda word: _check_stop_words(word, stop_words)).dropna()
    words_list = words_list.apply(lambda word: FSMStemmer().stem(words=word)[0])
    words_list = words_list.apply(_check_word_len).dropna().reset_index(drop=True)
    words_list = words_list.apply(lambda word: _check_stop_words(word, stop_words)).dropna()
    return words_list

def clean_text(text: str, stop_words: list) -> str:
    splited_text = ' '.join(re.findall(r"[A-Za-z 'ʻʼ.]+", text)).split(sep='.')
    clean_text = ""
    for sentence in splited_text:
        sentence_array = np.array([], dtype=object)
        sentence_word_list = sentence.split()
        for word in sentence_word_list:
            splited_word_list = _split_upper(word)
            sentence_array = np.append(sentence_array, splited_word_list)
        if sentence_array.shape[0] == 0:
            continue
        clean_sentence = _clean_pipline(sentence_array)
        clean_sentence_text = ' '.join(clean_sentence.values)
        if clean_sentence_text == '':
            continue
        clean_text += ' '.join(clean_sentence_text.split())
        clean_text += '. '
    return clean_text

def reshape_dataframe_by_dot_split(dataframe: pd.DataFrame, text_column: str) -> pd.DataFrame:
    result = pd.DataFrame(columns=['article_index', text_column])
    for i in tqdm(range(dataframe.shape[0])):
        lambda_ = lambda text: ' '.join(text.split())
        sentences = list(map(lambda_, dataframe.loc[i, text_column].split(sep='.')))
        for sentence in sentences:
            if sentence == '':
                continue
            result = result.append({'article_index': i,
                                    text_column: sentence}, ignore_index=True)
    return result


df_wiki_texts = pd.read_csv("TurkmenWikiTexts.csv")
stop_words = json.load(open("TurkmenStopWords.json")) + STOPWORDS_ADD

# Clean Texts

lambda_ = lambda text: clean_text(text, stop_words)
df_wiki_texts['clean_text'] = df_wiki_texts.loc[:, 'article'].apply(lambda_)
df_wiki_texts = df_wiki_texts.dropna()

# Since we need to get a corpus of ~10,000 documents, it is permissible that each document contains at least 130 words.

lambda_ = lambda text: len(text.split()) > 130
df_corpus = df_wiki_texts[df_wiki_texts['clean_text'].apply(lambda_)].reset_index(drop=True)
df_clean_corpus = df_corpus[['clean_text']]

df_corpus_dot_split = reshape_dataframe_by_dot_split(df_clean_corpus, 'clean_text')
df_corpus_dot_split.to_csv("TurkmenCleanCorpusDotSplit.csv", index=False)

df_uz_corpus = df_corpus_dot_split.groupby(by='article_index').agg({'clean_text': ' '.join}).reset_index()

df_uz_corpus.to_csv("TurkmenCleanCorpus.csv", index=False)
