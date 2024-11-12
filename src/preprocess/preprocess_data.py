from tqdm import tqdm

def preprocess_data(df_train, df_test, df_train_targets):
    """Preprocess the data."""
    # Get rid of the ID column
    df_train = df_train.iloc[:, 1:]
    df_test = df_test.iloc[:, 1:]
    df_train_targets = df_train_targets.iloc[:, 1:]
    
    # Standardize each column in the train and test sets with progress bars
    for col in tqdm(df_train.columns, desc="Standardizing Train Data"):
        df_train[col] = (df_train[col] - df_train[col].mean()) / df_train[col].std()
        
    for col in tqdm(df_test.columns, desc="Standardizing Test Data"):
        df_test[col] = (df_test[col] - df_test[col].mean()) / df_test[col].std()
    
    return df_train, df_test, df_train_targets

