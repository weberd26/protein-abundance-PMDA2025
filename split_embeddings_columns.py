#!/usr/bin/env python3
# split_embeddings_columns.py

"""
HOW TO USE:
    # Split sequence & network embeddings into separated columns:
    python split_embeddings_columns.py data/train_set2_embeddings.csv
    python split_embeddings_columns.py data/val_set2_embeddings.csv
    python split_embeddings_columns.py data/test_set2_embeddings.csv
"""

import argparse
import os
import ast
import numpy as np
import pandas as pd

def split_embeddings(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    # convert each string like "[0.1, -0.2, …]" into a 1D numpy array
    arrays = df[col].apply(
        lambda x: np.array(ast.literal_eval(x))
        if isinstance(x, str) else x
    )
    # stack into a 2D array
    matrix = np.vstack(arrays.values)
    # generate column names
    cols = [f"{prefix}_{i}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, index=df.index, columns=cols)

def main():
    parser = argparse.ArgumentParser(
        description="Split 'embeddings_sequence' & 'embeddings_network' into numeric columns"
    )
    parser.add_argument(
        "input_csv",
        help="Path to the CSV containing 'embeddings_sequence' and 'embeddings_network'"
    )
    parser.add_argument(
        "-o", "--output",
        help="Optional: specify output CSV path (default: input_split.csv)",
        default=None
    )
    args = parser.parse_args()

    # 1) Read the CSV (use the first column as index)
    df = pd.read_csv(args.input_csv, index_col=0)

    # 2) Split 'embeddings_sequence' into seq_0...seq_n
    seq_df = split_embeddings(df, "embeddings_sequence", "seq")
    # 3) Split 'embeddings_network'   into net_0...net_m
    net_df = split_embeddings(df, "embeddings_network",  "net")

    # 4) Drop the original embedding columns
    df = df.drop(columns=["embeddings_sequence", "embeddings_network"])

    # 5) Concatenate all parts horizontally
    out_df = pd.concat([df, seq_df, net_df], axis=1)

    # 6) Write to CSV
    base, _ = os.path.splitext(args.input_csv)
    output_path = args.output or f"{base}_split.csv"
    out_df.to_csv(output_path)
    print(f"Written split embeddings to: {output_path}")

if __name__ == "__main__":
    main()
