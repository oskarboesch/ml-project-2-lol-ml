import pandas as pd
from tqdm import tqdm

def preprocess_data(df_train, df_test, df_train_targets):
    """Preprocess the data."""
    # Drop the ID column
    df_train = df_train.iloc[:, 1:]
    df_test = df_test.iloc[:, 1:]
    df_train_targets = df_train_targets.iloc[:, 1:]
    
    # Ensure numeric data
    df_train = df_train.apply(pd.to_numeric, errors='coerce')
    df_test = df_test.apply(pd.to_numeric, errors='coerce')
    # Find columns that are empty in df_train or df_test
    zero_std_columns = df_train.columns[(df_train.std() == 0) | (df_test.std() == 0)]
    # Print the empty columns
    print(f"Empty columns: {zero_std_columns}")
    # Remove these columns from both datasets
    df_train = df_train.drop(columns=zero_std_columns)
    df_test = df_test.drop(columns=zero_std_columns)
    
    # Standardize columns with checks
    for col in tqdm(df_train.columns, desc="Standardizing Train Data"):
        if df_train[col].std() != 0:
            df_train[col] = (df_train[col] - df_train[col].mean()) / df_train[col].std()
        else:
            print(f"Skipping column '{col}' in train data: standard deviation is zero.")
    
    for col in tqdm(df_test.columns, desc="Standardizing Test Data"):
        if df_test[col].std() != 0:
            df_test[col] = (df_test[col] - df_test[col].mean()) / df_test[col].std()
        else:
            print(f"Skipping column '{col}' in test data: standard deviation is zero.")

    # Copy the train data
    df_train_categorical = df_train.copy()
    
    # Get tissue column from target and add it to train data
    df_train_categorical['tissue'] = df_train_targets['tissue']
    df_train_targets = df_train_targets.drop(columns='tissue')
    # One-hot encode the tissue column
    df_train_categorical = pd.get_dummies(df_train_categorical, columns=['tissue'])

    
    return df_train, df_train_categorical, df_test, df_train_targets


