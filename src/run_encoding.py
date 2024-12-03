from utils.utils import load_data, load_cgc_data, set_random_seed, load_pca_data
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils.vae import VAE, train_autoencoder, train_vae
import torch

# Constants
DATA_PATH = Path('data/encoded')
RESULTS_PATH = Path('results/figures')
RANDOM_SEED = 42

# Configuration
CONFIG = {
    'latent_dim': 100,
    'batch_size': 32,
    'epochs_autoencoder': 20,
    'epochs_vae': 10,
    'lr_vae': 1e-2
}

# Set random seed for reproducibility
set_random_seed(RANDOM_SEED)


def load_and_split_data(data_type='normal'):
    """
    Loads and splits the dataset based on the specified type.

    Args:
        data_type (str): Type of data to load ('normal' or 'cgc').

    Returns:
        Tuple: X_train, X_test, y_train, y_test, total_data
    """
    if data_type == 'normal':
        train_data, test_data, train_target = load_data(raw=False, categorical=False)
    elif data_type == 'cgc':
        train_data, test_data, train_target = load_cgc_data()
    elif data_type == 'pca':
        train_data, test_data, train_target = load_pca_data()
    else:
        raise ValueError("Invalid data_type. Choose 'normal', 'cgc' or 'pca'")

    X_train, X_test, y_train, y_test = train_test_split(
        train_data, train_target, test_size=0.2, random_state=RANDOM_SEED
    )
    total_data = pd.concat([train_data, test_data])
    return X_train, X_test, y_train, y_test, total_data


def evaluate_linear_model(X_train, X_test, y_train, y_test):
    """
    Trains a linear regression model and evaluates it.

    Args:
        X_train (array-like): Training features.
        X_test (array-like): Test features.
        y_train (array-like): Training labels.
        y_test (array-like): Test labels.

    Returns:
        Tuple: MSE, R², Spearman correlation
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    return mse, r2, spearman_corr


def save_plot(latent_features, y_train, title, save_path):
    """
    Saves a scatter plot of latent features.

    Args:
        latent_features (array): Latent features to plot.
        y_train (array): Labels for coloring the plot.
        title (str): Plot title.
        save_path (Path): Path to save the plot.
    """
    plt.scatter(latent_features[:, 0], latent_features[:, 1], c=y_train['AAC'], cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def process_data(data_type):
    """
    Processes the specified data type with autoencoder and VAE.

    Args:
        data_type (str): Type of data to process ('normal' or 'cgc').
    """
    print(f"Processing {data_type.upper()} data...")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, total_data = load_and_split_data(data_type)
    input_dim = X_train.shape[1]

    # Train Autoencoder
    encoder = train_autoencoder(X_train, X_test, input_dim, CONFIG)
    encoded_features_train = encoder.predict(X_train)
    encoded_features_test = encoder.predict(X_test)

    # Evaluate Autoencoder
    mse, r2, spearman = evaluate_linear_model(encoded_features_train, encoded_features_test, y_train, y_test)
    print(f"--- {data_type.upper()} Autoencoder Results ---")
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}, Spearman's ρ: {spearman:.4f}")

    # Save AE plot and data
    save_plot(encoded_features_train, y_train, f"Spearman's Correlation: {spearman:.4f}",
              RESULTS_PATH / f'ae_latent_features_{data_type}.png')
    encoded_features_total = encoder.predict(total_data)
    pd.DataFrame(
        encoded_features_total,
        columns=[f"encoded_feature_{i}" for i in range(encoded_features_total.shape[1])]
    ).to_csv(DATA_PATH / f'ae_data_{data_type}.csv', index=False)

    # Train VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = train_vae(X_train, X_test, CONFIG, device)

    # Evaluate VAE
    latent_train, _ = vae_model.encode(torch.tensor(X_train.values, dtype=torch.float32).to(device))
    latent_test, _ = vae_model.encode(torch.tensor(X_test.values, dtype=torch.float32).to(device))
    latent_train = latent_train.cpu().detach().numpy()
    latent_test = latent_test.cpu().detach().numpy()

    encoded_features_total_vae, _ = vae_model.encode(torch.tensor(total_data.values, dtype=torch.float32).to(device))

    mse_vae, r2_vae, spearman_vae = evaluate_linear_model(latent_train, latent_test, y_train, y_test)
    print(f"--- {data_type.upper()} VAE Results ---")
    print(f"MSE: {mse_vae:.4f}, R²: {r2_vae:.4f}, Spearman's ρ: {spearman_vae:.4f}")

    # Save VAE plot and data
    save_plot(latent_train, y_train, f"Best Spearman's Correlation: {spearman_vae:.4f}",
              RESULTS_PATH / f'vae_latent_features_{data_type}.png')
    pd.DataFrame(
        encoded_features_total_vae.cpu().detach().numpy(),
        columns=[f"latent_feature_{i}" for i in range(encoded_features_total_vae.shape[1])]
    ).to_csv(DATA_PATH / f'vae_data_{data_type}.csv', index=False)


# ------------------------------
# Main Script
# ------------------------------
if __name__ == "__main__":
    for data_type in ['normal', 'cgc', 'pca']:
        process_data(data_type)
