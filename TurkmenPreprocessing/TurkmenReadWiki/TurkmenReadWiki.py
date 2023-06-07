"""

Read Turkmen Wiki

"""

import re
import glob
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from uuid import uuid4

# nltk.download('punkt')


def split_keep_sep(string: str, sep: str) -> list:
    cleaned = []
    string = re.split('(%s)' % re.escape(sep), string)
    for _ in string:
        if _ != '' and _ != sep:
            cleaned.append(sep + _)
    return cleaned

def remove_html_tags(text: str) -> str:
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_special_chars(text: str, char_list: list) -> str:
    for char in char_list:
        text = text.replace(char, '')
    return text.replace(u'\xa0', u' ')

def process_wiki_file(wiki_file: str) -> pd.DataFrame:
    chars = ['\n']
    with open(wiki_file, encoding='utf-8') as f:
        content = f.read()
    articles = split_keep_sep(content, '<doc id=')
    dataframe = pd.DataFrame(columns=['article'])
    for num, article in enumerate(articles):
        article = remove_special_chars(remove_html_tags(article), chars)
        if len(article.split()) < 50:
            continue
        dataframe = dataframe.append({'article': article}, ignore_index=True)
    return dataframe


wiki_files = []
for filename in glob.iglob("TurkmenWiki/*/*"):
    wiki_files.append(filename)

dataframe = pd.DataFrame()
for file_name in tqdm(wiki_files):
    dataframe_file = process_wiki_file(file_name)
    dataframe = pd.concat([dataframe, dataframe_file])
dataframe.reset_index(drop=True, inplace=True)

dataframe.to_csv("TurkmenWikiTexts.csv", index=False)
