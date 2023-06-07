"""

Group the data by page (pg_index)

"""

import numpy as np
import pandas as pd

PATH = "EnWikiTexts.csv"


def group_texts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df_page_ind = pd.DataFrame(df['article_uuid'].unique(), columns=['article_uuid'])
    df_page_ind = df_page_ind.reset_index().rename(columns={'index': 'pg_index'})
    df = df.merge(df_page_ind, how='left', on='article_uuid')
    df = df[['proc_sentence', 'pg_index']]
    df = df.astype({'proc_sentence': str}).groupby(by='pg_index').agg({'proc_sentence': ' '.join})
    df_pg = df.reset_index()
    df_pg = df_pg.dropna()
    return df_pg


group_texts("EnWikiTexts.csv").to_csv("EnGroupedTexts.csv", index=False)
