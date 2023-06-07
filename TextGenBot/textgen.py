'''
Char-based text generation with LSTM
'''

from collections import Counter
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

def evaluate(model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)

    inp = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char

    return predicted_text


def generate_text(df_texts: pd.DataFrame,
                  split_range: int = 100,
                  start_length: int = 10,
                  prediction_length: int = 750) -> pd.DataFrame:
    df_bot = pd.DataFrame(columns=['predicted_text'])
    file_ =
    for i in tqdm(range(df_texts.shape[0])):
        text = df_texts.loc[i, 'preprocessed_text']
        splitted_text = text.split()
        predicted_text = ""
        for j in range(0, len(splitted_text), split_range):
            fragment = splitted_text[j:j+split_range]
            start_text = ' '.join(fragment[:start_length])
            predicted_fragment = evaluate(model,
                                          char_to_idx,
                                          idx_to_char,
                                          temp=0.3,
                                          prediction_len=prediction_length,
                                          start_text=start_text + ' ')
            predicted_fragment = ' '.join(predicted_fragment.split()[start_length:-1])
            predicted_text += predicted_fragment
            predicted_text += ' '
        df_bot.loc[i, 'predicted_text'] = predicted_text
    return df_bot



model_path = sys.argv[1]
df_texts_path = sys.argv[2]
k = sys.argv[3]

model = torch.load(model_path)
df_texts = pd.read_csv(df_texts_path)
df_bot = generate_text(df_texts)
df_bot.to_csv(f"{k}_df_bot.csv", index=False)
