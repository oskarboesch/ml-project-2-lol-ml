import pandas as pd
import sys
from pathlib import Path
import numpy as np
import os


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

def load_encoded_data():
    """Load the encoded data from the preprocessed folder."""
    # Add the project root directory to the Python path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    data_path = project_root / 'data' / 'encoded'
    return pd.read_csv(data_path / 'ae_data.csv'), pd.read_csv(data_path / 'vae_data.csv'), pd.read_csv(data_path / 'vae2_data.csv')

def save_data(df_train, df_train_categorical, df_test, df_train_targets, df_total):
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
    print("Saving preprocessed data to tsv...")
    # Save in tsv format as well
    df_train.to_csv(data_path / 'train.tsv', sep='\t', index=False)
    df_train_categorical.to_csv(data_path / 'train_categorical.tsv', sep='\t', index=False)
    df_test.to_csv(data_path / 'test.tsv', sep='\t', index=False)
    df_train_targets.to_csv(data_path / 'train_targets.tsv', sep='\t', index=False)
    df_total.to_csv(data_path / 'total_data.tsv', sep='\t', index=False)
    print("Preprocessed data saved.")


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