def parse_embeddings(col):
    return np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in col])

if 'embeddings_sequence' in df_train.columns:
    df_train['embeddings_sequence'] = parse_embeddings(df_train['embeddings_sequence'])
    df_val['embeddings_sequence'] = parse_embeddings(df_val['embeddings_sequence'])
    df_test['embeddings_sequence'] = parse_embeddings(df_test['embeddings_sequence'])
