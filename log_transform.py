organs = [col for col in df_train.columns if not col.startswith("pca_") and col != target_column]

pca_cols = [col for col in df_train.columns if col.startswith('pca_')]

# Log transform organ columns only
X_train_organs = np.log1p(df_train[organs].values)
X_val_organs = np.log1p(df_val[organs].values)
X_test_organs = np.log1p(df_test[organs].values)

# PCA columns as is
X_train_pca = df_train[pca_cols].values
X_val_pca = df_val[pca_cols].values
X_test_pca = df_test[pca_cols].values

# Combine organs and PCA columns horizontally
X_train = np.hstack([X_train_organs, X_train_pca])
X_val = np.hstack([X_val_organs, X_val_pca])
X_test = np.hstack([X_test_organs, X_test_pca])

# Target with log1p
y_train = np.log1p(df_train[target_column].values)
y_val = np.log1p(df_val[target_column].values)
y_test = np.log1p(df_test[target_column].values)
