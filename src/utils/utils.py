import pandas as pd
import sys
from pathlib import Path

def load_data(raw = True):
    """Load train, test and train targets data from  CSV files."""
    # Add the project root directory to the Python path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    data_path = project_root / 'data'
    if raw:
        folder_path = data_path / 'raw'
    else:
        folder_path = data_path / 'preprocessed'
    return pd.read_csv(folder_path / 'train.csv'), pd.read_csv(folder_path / 'test.csv'), pd.read_csv(folder_path / 'train_targets.csv')

def save_data(df_train, df_test, df_train_targets):
    """Save the preprocessed train, test and train targets data to a CSV file."""
    # Add the project root directory to the Python path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    data_path = project_root / 'data' / 'preprocessed'
    df_train.to_csv(data_path / 'train.csv', index=False)
    df_test.to_csv(data_path / 'test.csv', index=False)
    df_train_targets.to_csv(data_path / 'train_targets.csv', index=False)