import pandas as pd
import sys
from pathlib import Path
import numpy as np
import os
import torch
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler



def load_data(raw = True, categorical = False):
    """Load train, test and train targets data from  CSV files."""
    # Add the project root directory to the Python path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    data_path = project_root / 'data'
    if raw:
        folder_path = data_path / 'raw'
    else:
        folder_path = data_path / 'preprocessed'
    if categorical:
        train_path = folder_path / 'train_categorical.csv'
    else:
        train_path = folder_path / 'train.csv'
    return pd.read_csv(train_path), pd.read_csv(folder_path / 'test.csv'), pd.read_csv(folder_path / 'train_targets.csv')

def load_encoded_data(data_type='normal'):
    """
    Load the encoded data from the preprocessed folder based on the specified data type.

    Parameters:
        data_type (str): The type of data to load. Options are 'normal', 'pca', or 'cgc'.

    Returns:
        tuple: A tuple containing three DataFrames corresponding to AE, VAE, and VAE2 data.
    """
    # Add the project root directory to the Python path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    data_path = project_root / 'data' / 'encoded'
    
    # Define file mappings based on data_type
    file_mapping = {
        'normal': {
            'ae': 'ae_data_normal.csv',
            'vae': 'vae_data_normal.csv',
            'vae2': 'vae2_data_normal.tsv',
        },
        'pca': {
            'ae': 'ae_data_pca.csv',
            'vae': 'vae_data_pca.csv',
            'vae2': 'vae2_data_pca.tsv',
        },
        'cgc': {
            'ae': 'ae_data_cgc.csv',
            'vae': 'vae_data_cgc.csv',
            'vae2': 'vae2_data_cgc.tsv',
        },
    }
    
    if data_type not in file_mapping:
        raise ValueError("Invalid data_type specified. Choose from 'normal', 'pca', or 'cgc'.")
    
    # Load datasets
    ae_data = pd.read_csv(data_path / file_mapping[data_type]['ae'])
    vae_data = pd.read_csv(data_path / file_mapping[data_type]['vae'])
    if data_type != 'pca':
        vae2_data = pd.read_csv(data_path / file_mapping[data_type]['vae2'], sep='\t').iloc[:, 1:]
    else:
        vae2_data = None
    
    return ae_data, vae_data, vae2_data

def save_data(df_train, df_train_categorical, df_test, df_train_targets, df_total, cgc_train, cgc_test, pca_train, pca_test):
    """Save the preprocessed train, test and train targets data to a CSV file."""
    # Add the project root directory to the Python path
    print("Saving preprocessed data to csv...")
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    data_path = project_root / 'data' / 'preprocessed'
    df_train.to_csv(data_path / 'train.csv', index=False)
    df_train_categorical.to_csv(data_path / 'train_categorical.csv', index=False)
    df_test.to_csv(data_path / 'test.csv', index=False)
    df_train_targets.to_csv(data_path / 'train_targets.csv', index=False)
    df_total.to_csv(data_path / 'total_data.csv', index=False)
    cgc_train.to_csv(data_path / 'cgc_train.csv', index=False)
    cgc_test.to_csv(data_path / 'cgc_test.csv', index=False)
    cgc_total = pd.concat([cgc_train, cgc_test], axis=0)
    cgc_total.to_csv(data_path / 'cgc_total.csv', index=False)
    pca_train.to_csv(data_path / 'pca_train.csv', index=False)
    pca_test.to_csv(data_path / 'pca_test.csv', index=False)
    pca_total = pd.concat([pca_train, pca_test], axis=0)
    pca_total.to_csv(data_path / 'pca_total.csv', index=False)
    print("Preprocessed data saved to csv.")

def save_to_tsv(df_total, cgc_train, cgc_test, pca_train, pca_test):
    print("Saving preprocessed data to tsv...")
    # Save in tsv format as well
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    data_path = project_root / 'data' / 'preprocessed' 

    cgc_total = pd.concat([cgc_train, cgc_test], axis=0)
    pca_total = pd.concat([pca_train, pca_test], axis=0)

    df_total.to_csv(data_path / 'total_data.tsv', sep='\t', index=False)
    cgc_total.to_csv(data_path / 'cgc_total.tsv', sep='\t', index=False)
    pca_total.to_csv(data_path / 'pca_total.tsv', sep='\t', index=False)
    print("Preprocessed data saved to tsv.")



def get_panda_from_txt(file_path, train_data):
    """Placeholder function to process extra data files."""
    extra_data = pd.read_csv(file_path, sep='\t', index_col=0)
    extra_data = extra_data.apply(pd.to_numeric, errors='coerce').astype(np.float64)

    extra_data = extra_data.T
    extra_data = extra_data.iloc[:, 2:]
    # Get rid of first two rows
    extra_data = extra_data.iloc[3:]

    # Standardize the features and skip std=0 columns
    std_devs = extra_data.std()
    non_zero_std_cols = std_devs[std_devs != 0].index
    extra_data = extra_data[non_zero_std_cols]
    extra_data = (extra_data - extra_data.mean()) / extra_data.std()

    # Get common columns
    common_columns = extra_data.columns.intersection(train_data.columns)
    print(f'Common columns: {common_columns}')
    # Only keep common columns
    extra_data = extra_data[common_columns]
    # Add empty columns for the missing columns from train to extra_data
    extra_data = extra_data.reindex(columns=train_data.columns, fill_value=0)

    return extra_data

def concatenate_all_dfs(df_train, gene_expression_datasets):
    # List to store newly loaded DataFrames
    new_dataframes = []
    # Extract all the txt files of each folder to pandas dataframes
    # This is a placeholder for the next step of the pipeline
    # The next step will be to preprocess the data
    # and save it to the preprocessed folder
    for dataset, _ in gene_expression_datasets:
       dataset_folder = os.path.join('data', 'extra_data', dataset)
       for file in os.listdir(dataset_folder):
              if file.endswith('.txt'):
                print(f'Processing {dataset_folder}')
                # Load the file into a pandas dataframe
                df = get_panda_from_txt(os.path.join(dataset_folder, file), df_train)
                # Check mean and std of the new data
                new_dataframes.append(df)
    # Concatenate all the new DataFrames into one
    if new_dataframes:
        concatenated_new_df = pd.concat(new_dataframes, ignore_index=True)
        # Concatenate with the existing df_train
        df_train = pd.concat([df_train, concatenated_new_df], ignore_index=True)

    print("All new data concatenated with x_train.")

    # Save the new df_train to the preprocessed folder
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    data_path = project_root / 'data' / 'preprocessed'
    print("Saving new data...")
    df_train.to_csv(data_path / 'train_augmented.csv', index=False)
    print("New data saved.")

def load_augmented_data():
    # Add the project root directory to the Python path
    data_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'preprocessed'
    # Load the augmented data
    train_augmented = pd.read_csv(data_path / 'train_augmented.csv')
    percentage_zeros = (train_augmented == 0).mean()
    # Get all columns with more than 60% zeros
    columns_to_drop = percentage_zeros[percentage_zeros > 0.6].index
    # drop these columns
    train_augmented = train_augmented.drop(columns=columns_to_drop)
    return train_augmented

def load_cgc_data():
    # Add the project root directory to the Python path
    data_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'preprocessed'
    # Load the CGC data
    cgc_train = pd.read_csv(data_path / 'cgc_train.csv')
    cgc_test = pd.read_csv(data_path / 'cgc_test.csv')
    train_targets = pd.read_csv(data_path / 'train_targets.csv')
    return cgc_train, cgc_test, train_targets

def load_pca_data():
    # Add the project root directory to the Python path
    data_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'preprocessed'
    # Load the PCA data
    pca_train = pd.read_csv(data_path / 'pca_train.csv')
    pca_test = pd.read_csv(data_path / 'pca_test.csv')
    train_targets = pd.read_csv(data_path / 'train_targets.csv')
    return pca_train, pca_test, train_targets

def set_random_seed(seed = 42):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Ensure TensorFlow uses deterministic behavior (if using GPU)
    tf.config.experimental.enable_op_determinism()

def min_max_scale(df):
    """Scale the data using MinMaxScaler."""
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Fit the scaler on the training data
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df