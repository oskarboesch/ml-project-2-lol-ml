from utils.utils import load_data
import pandas as pd
from pathlib import Path  

DATA_PATH = Path(__file__).resolve().parent.parent / 'data'

if __name__ == "__main__":
    # HyperParameters
    input_dim = 50
    latent_dim = 10
    batch_size = 64
    epochs = 20

    # Load data
    print("Loading Data...")
    train, test, y = load_data(raw = False)
    train_augmented =  pd.read_csv(DATA_PATH / 'preprocessed' / 'train_augmented.csv')
    print("Data Loaded Successfully")
    