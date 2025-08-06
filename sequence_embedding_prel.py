"""
Author: TS
"""

import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
import re
import torch


# %% Get sequences

path_data = "../Organ_organsismDEF_PROCESSED_all_organisms_filtered_without_M.musculus_KIDNEY.csv"

df = pd.read_csv(path_data)   # adjust the path

sequences_raw = df['Sequence'].tolist() # deesired format ["A E T C Z A O","S K T Z P"]
sequences_in = [' '.join(w) for w in sequences_raw]  # ['a b c', 'd e']

# REMOVE LATER
sequences_in = sequences_in[:10]
###


# %% Compute embeddings

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")

encoded_input = tokenizer(sequences_in[0], return_tensors='pt')
output = model(**encoded_input)[0].detach().numpy()








