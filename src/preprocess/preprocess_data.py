import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def preprocess_data(df_train, df_test, df_train_targets, min_max_scale=False):
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
    if min_max_scale:
        scaler = MinMaxScaler()
        message = "Min-max scaling data selected for VAE2..."
    else :
        scaler = StandardScaler()
        message = "Standardizing data..."
    scaler.fit(df_train)
    print(message)
    df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns)
    if min_max_scale:
        scaler.fit(df_test)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

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


def create_CGC_data(df_train, df_test):
    """Create the CGC data."""
    # Get the CGC data
    cgc_data = pd.read_csv('data/CGC.csv')
    cgc_data = cgc_data['Gene Symbol']

    # Get the common columns
    common_columns = cgc_data[cgc_data.isin(df_train.columns)]
    # Get the common columns in the train data
    cgc_train = df_train[common_columns]
    # Get the common columns in the test data
    cgc_test = df_test[common_columns]
    

    return cgc_train, cgc_test

def create_PCA_data(df_train, df_test):
    """Create the PCA data."""
    pca = PCA()
    pca.fit(df_train)
    pca_train = pd.DataFrame(pca.transform(df_train))
    pca_test = pd.DataFrame(pca.transform(df_test))
    
    return pca_train, pca_test