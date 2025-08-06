"""
Author: TS
"""

import numpy as np
import pandas as pd
from protlearn.features import aac, ctdc, ctdt, ctdd, length

# %% Get sequences

path_data = "../Organ_organsismDEF_PROCESSED_all_organisms_filtered_without_M.musculus_KIDNEY.csv"

df = pd.read_csv(path_data)   # adjust the path

sequences_all = df['Sequence'].tolist() 

# # REMOVE LATER----
# sequences_all = sequences_all[:1000]
# ###---------

def featurize_protlearn(seqs, ids=None):
    # protlearn accepts a list of strings; we’ll uppercase and keep standard AAs
    AA = set("ACDEFGHIKLMNPQRSTVWY")
    clean = ["".join(ch for ch in s.upper() if ch in AA) for s in seqs]
    if ids is None:
        ids = [f"seq_{i}" for i in range(len(clean))]

    A, A_names = aac(clean)     # (n,20)
    C, C_names = ctdc(clean)    # (n,21)
    T, T_names = ctdt(clean)    # (n,21)
    D, D_names = ctdd(clean)    # (n,105)
    L              = length(clean)  # (n,1)

    X = np.concatenate([A, C, T, D, L], axis=1)
    cols = list(A_names) + list(C_names) + list(T_names) + list(D_names) + ["length"]
    return pd.DataFrame(X, columns=cols, index=ids)

# Example
df_feat = featurize_protlearn(sequences_all)






