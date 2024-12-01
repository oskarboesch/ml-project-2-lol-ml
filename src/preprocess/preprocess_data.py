import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

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
    
    # Standardize the data with a model
    scaler = MinMaxScaler()
    scaler.fit(df_train)
    print("Standardizing data...")
    df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns)
    scaler.fit(df_test)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)
    print("Data standardized.")

    # Copy the train data
    df_train_categorical = df_train.copy()
    
    # Get tissue column from target and add it to train data
    df_train_categorical['tissue'] = df_train_targets['tissue']
    df_train_targets = df_train_targets.drop(columns='tissue')
    # One-hot encode the tissue column
    df_train_categorical = pd.get_dummies(df_train_categorical, columns=['tissue'])

    # create a concatenation of train and test
    df_total = pd.concat([df_train, df_test], axis=0)

    
    return df_train, df_train_categorical, df_test, df_train_targets, df_total


