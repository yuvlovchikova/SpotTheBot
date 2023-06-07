import re
import sys
import glob
import nltk
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from uuid import uuid4
from functools import reduce
from multiprocessing import Pool

nltk.download('punkt')


def _remove_non_printed_chars(string):
    "leave only English letters"
    reg = re.compile('[^a-zA-ZäöåÄÖÅ]')
    return reg.sub(' ', string)

def _trim_string(string):
    "remove extra spaces, remove trailing spaces, lower the case"
    return re.sub('\s+', ' ', string).strip().lower()

def clean_string(string):
    string = _remove_non_printed_chars(string)
    string = _trim_string(string)
    return string

def split_keep_sep(string, sep):
    cleaned = []
    string = re.split("(%s)" % re.escape(sep), string)
    for _ in string:
        if _ != '' and _ != sep:
            cleaned.append(sep + _)
    return cleaned

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_special_chars(text, char_list):
    for char in char_list:
        text = text.replace(char, '')
    return text.replace(u'\xa0', u' ')

def process_wiki_files(wiki_file):
    chars = ['\n']
    with open(wiki_file, encoding='utf-8') as f:
        content = f.read()
    articles = split_keep_sep(content, '<doc id=')
    df = pd.DataFrame(columns=['article_uuid', 'sentence', 'proc_sentence', 'proc_len'])
    for article in articles:
        uuid = uuid4()
        article = remove_special_chars(remove_html_tags(article), chars)
        sentences = nltk.sent_tokenize(article)
        proc_sentences = [clean_string(sentence) for sentence in sentences]
        proc_lens = [len(sentence.split(' ')) for sentence in proc_sentences]
        temp_df = pd.DataFrame({'article_uuid': [uuid] * len(sentences),
                                'sentence': sentences,
                                'proc_sentence': proc_sentences,
                                'proc_len': proc_lens})
        df = df.append(temp_df)
    return df

def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params, **kwargs)

def list_multiprocessing(param_lst, func, **kwargs):
    if __name__ == '__main__':
        workers = kwargs.pop('workers')
        with Pool(workers) as p:
            apply_lst = [([params], func, i, kwargs) for i, params in enumerate(param_lst)]
            result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))
        result=sorted(result, key=lambda x: x[0])
        return [_[1] for _ in result]


wiki_files = []
for filename in glob.iglob("EnWiki/*/*", recursive=True):
    wiki_files.append(filename)


df = list_multiprocessing(wiki_files, process_wiki_files, workers=4)
df = pd.concat(df).reset_index(drop=True)
df.article_uuid = df.article_uuid.astype(str)
df.to_csv("EnWikiTexts.csv", index=False)
