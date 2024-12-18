import torch
import torch.nn as nn
import itertools
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid, train_test_split, KFold
from utils import set_random_seed


FIGURES_PATH = '../../results/figures/'
#Generate pairs for MarginRankingLoss
def generate_pairs(X, y):
    """
    Generate all pairwise combinations of data and corresponding ranking labels.
    Args:
        X: Features (torch.Tensor)
        y: Targets (torch.Tensor)
    Returns:
        X1, X2: Pairwise feature tensors
        y_pairs: Pairwise ranking labels (+1 or -1)
    """
    set_random_seed(42)
    pairs = list(itertools.combinations(range(len(y)), 2))  # Generate index pairs
    X1 = torch.stack([X[i] for i, j in pairs])
    X2 = torch.stack([X[j] for i, j in pairs])
    y_pairs = torch.tensor([1 if y[i] > y[j] else -1 for i, j in pairs], dtype=torch.float32)
    return X1, X2, y_pairs


class RankMLP(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, hidden_layers=(64, 32), learning_rate=1e-3, weight_decay=1e-5, margin=0.5, epochs=100, patience=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.margin = margin
        self.epochs = epochs
        self.patience = patience
        self.model = None

    def _build_model(self):
        layers = []
        current_size = self.input_size
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
            current_size = hidden_size
        layers.append(nn.Linear(current_size, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y, X_val, y_val, verbose=False):
        set_random_seed(42)
        # Prepare training data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        X1, X2, y_pairs = generate_pairs(X_tensor, y_tensor)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        # Initialize model, optimizer, and loss function
        self.model = self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MarginRankingLoss(margin=self.margin)

        # Early stopping variables
        best_val_spearman = -np.inf
        best_model = None
        epochs_no_improve = 0

        # Metrics storage
        train_spearman_list = []
        val_spearman_list = []
        train_mse_list = []
        val_mse_list = []

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            outputs1 = self.model(X1).squeeze()
            outputs2 = self.model(X2).squeeze()
            loss = criterion(outputs1, outputs2, y_pairs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                train_predictions = self.model(X_tensor).squeeze().numpy()
                train_spearman, _ = spearmanr(train_predictions, y)
                train_mse = mean_squared_error(train_predictions, y)
                train_spearman_list.append(train_spearman)
                train_mse_list.append(train_mse)

                val_predictions = self.model(X_val).squeeze().numpy()
                val_spearman, _ = spearmanr(val_predictions, y_val)
                val_mse = mean_squared_error(val_predictions, y_val)
                val_spearman_list.append(val_spearman)
                val_mse_list.append(val_mse)

                # Early stopping logic
                if val_spearman > best_val_spearman:
                    best_val_spearman = val_spearman
                    best_model = self.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

                if epoch % 10 == 0 and verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - Train Spearman: {train_spearman:.4f} - Validation Spearman: {val_spearman:.4f}")

        # Restore the best model
        self.model.load_state_dict(best_model)
        if verbose:
            print(f"Best validation Spearman: {best_val_spearman:.4f}")


        return train_spearman_list, val_spearman_list, train_mse_list, val_mse_list, best_val_spearman

    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
    
def grid_search(data_train, label_train, data_test, label_test, param_grid, cv=3, evaluate=True, verbose=False):
    # Store all results in a list of dictionaries
    all_results = []
    params_best_spearman = []

    for params in ParameterGrid(param_grid):
        if verbose:
            print(f"Training with parameters: {params}")

        # Initialize KFold cross-validation
        kf = KFold(n_splits=cv, random_state=42, shuffle=True)

        fold_best_spearman = []
        fold_results = []  # Store results per fold
        for fold_idx, (train_index, val_index) in enumerate(kf.split(data_train)):
            # Split train and validation data
            X_train, X_val = data_train[train_index], data_train[val_index]
            y_train, y_val = label_train[train_index], label_train[val_index]

            set_random_seed(42)
            # Set model parameters
            model = RankMLP(data_train.shape[1])
            model.set_params(**params)

            # Train the model and return metrics
            train_spearman, val_spearman, train_mse, val_mse, best_spearman = model.fit(X_train, y_train, X_val, y_val)

            # Append results for this fold
            fold_results.append({
                'params': params,
                'fold': fold_idx + 1,
                'train_spearman': train_spearman,
                'val_spearman': val_spearman,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'best_spearman': best_spearman
            })

            if verbose:
                print(f"Fold {fold_idx + 1}/{cv}: Val Spearman: {best_spearman:.4f}")
            
            fold_best_spearman.append(best_spearman)

        params_best_spearman.append(np.mean(fold_best_spearman))


        # Add results to the master list
        all_results.extend(fold_results)

    # Convert all results to a DataFrame for easy plotting
    results_df = pd.DataFrame(all_results)

    # Find the best parameter set based on average Spearman score
    best_params = list(ParameterGrid(param_grid))[np.argmax(params_best_spearman)]
    # Evaluate the model on the test set using the best parameters
    if evaluate:
        test_spearman, test_mse = evaluate_model(model, data_train, label_train, data_test, label_test, best_params)

    return results_df, best_params, test_spearman, test_mse

def evaluate_model(model, data_train,label_train, data_test, label_test, best_params):
    print("--- Evaluating model on test set...")
    set_random_seed(42)
    # Final Evaluation
    model.set_params(**best_params)
    # Define val for early stopping
    X_train, X_val, y_train, y_val = train_test_split(data_train, label_train, test_size=0.4, random_state=42)
    model.fit(X_train, y_train, X_val, y_val, verbose=True)

    # evaluate on test set
    y_pred = model.predict(data_test)
    test_spearman, _ = spearmanr(y_pred, label_test)
    test_mse = mean_squared_error(y_pred, label_test)
    print(f"With parameters : {best_params} - Test Spearman: {test_spearman:.4f} - Test MSE: {test_mse:.4f}")
    return test_spearman, test_mse


def plot_results(results_df, param_grid, title = 'RankMLP Validation Results', big_font = False):
    # Generate all combinations of the parameter grid
    param_combinations = list(itertools.product(*param_grid.values()))

    # Generate a color map based on the number of unique parameter combinations
    color_map = plt.cm.plasma(np.linspace(0, 1, len(param_combinations)))

    # Plot Spearman scores
    plt.figure(figsize=(12, 6))
    if big_font:
        plt.rcParams.update({
        'font.size': 20,  # General font size
        'axes.titlesize': 24,  # Title font size
        'axes.labelsize': 22,  # Axis label font size
        'xtick.labelsize': 20,  # X-axis tick label font size
        'ytick.labelsize': 20,  # Y-axis tick label font size
        'legend.fontsize': 18,  # Legend font size
        'legend.title_fontsize': 20,  # Legend title font size
        })  
    else :
        plt.title(title + ' - Spearman')

    # Iterate over each parameter combination and its index
    for param_index, param_combination in enumerate(param_combinations):
        # Convert tuple to dictionary for the parameter combination
        param_tuple_dict = {
            'hidden_layers': param_combination[0],
            'learning_rate': param_combination[1],
            'weight_decay': param_combination[2],
            'epochs': param_combination[3],
            'patience': param_combination[4]
        }

        color = color_map[param_index]

        # Filter data for the current parameter combination
        param_data = results_df[results_df['params'].apply(lambda x: x == param_tuple_dict)]

        # Iterate over the folds and plot the corresponding data for each parameter combination
        for fold in param_data['fold'].unique():
            fold_data = param_data[param_data['fold'] == fold]

            epochs = np.arange(1, len(fold_data['train_spearman'].values[0]) + 1)
            plt.plot(np.log(epochs), fold_data['train_spearman'].values[0], color=color, linestyle='-', alpha=0.7)
            plt.plot(np.log(epochs), fold_data['val_spearman'].values[0], color=color, linestyle='--', alpha=0.7)

    # Add a custom legend with labels for train and validation
    train_patch = plt.Line2D([0], [0], color='black', linestyle='-', alpha=0.7, label='Train')
    val_patch = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.7, label='Validation')

    # Create legend handles for parameter combinations (one handle per parameter combination)
    param_legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for color in color_map]
    param_legend_labels = [str(param_comb) for param_comb in param_combinations]

    # Combine both train/validation and parameter combinations in the legend
    plt.legend(handles=[train_patch, val_patch] + param_legend_handles,
               labels=['Train', 'Validation'] + param_legend_labels,
               title='Parameter Combinations', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel('Log(Epoch)')
    plt.ylabel('Spearman')
    plt.savefig(FIGURES_PATH + title + ' - Spearman.png')
    plt.show()

    # Plot MSE scores
    plt.figure(figsize=(12, 6))

    if big_font:
        plt.rcParams.update({
        'font.size': 20,  # General font size
        'axes.titlesize': 24,  # Title font size
        'axes.labelsize': 22,  # Axis label font size
        'xtick.labelsize': 20,  # X-axis tick label font size
        'ytick.labelsize': 20,  # Y-axis tick label font size
        'legend.fontsize': 18,  # Legend font size
        'legend.title_fontsize': 20,  # Legend title font size
        })  
    else :
        plt.title(title + ' - MSE')

    # Iterate over each parameter combination and its index (same code as before)
    for param_index, param_combination in enumerate(param_combinations):
        # Convert tuple to dictionary for the parameter combination
        param_tuple_dict = {
            'hidden_layers': param_combination[0],
            'learning_rate': param_combination[1],
            'weight_decay': param_combination[2],
            'epochs': param_combination[3],
            'patience': param_combination[4]
        }

        color = color_map[param_index]

        # Filter data for the current parameter combination
        param_data = results_df[results_df['params'].apply(lambda x: x == param_tuple_dict)]

        # Iterate over the folds and plot the corresponding data for each parameter combination
        for fold in param_data['fold'].unique():
            fold_data = param_data[param_data['fold'] == fold]

            # Plot training and validation MSE scores
            epochs = np.arange(1, len(fold_data['train_spearman'].values[0]) + 1)
            plt.plot(np.log(epochs), fold_data['train_mse'].values[0], color=color, linestyle='-', alpha=0.7)
            plt.plot(np.log(epochs), fold_data['val_mse'].values[0], color=color, linestyle='--', alpha=0.7)


    # Add a custom legend with labels for train and validation
    mse_train_patch = plt.Line2D([0], [0], color='black', linestyle='-', alpha=0.7, label='Train')
    mse_val_patch = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.7, label='Validation')

    # Create legend handles for parameter combinations (one handle per parameter combination)
    mse_param_legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for color in color_map]
    mse_param_legend_labels = [str(param_comb) for param_comb in param_combinations]

    # Combine both train/validation and parameter combinations in the legend
    plt.legend(handles=[mse_train_patch, mse_val_patch] + mse_param_legend_handles,
               labels=['Train', 'Validation'] + mse_param_legend_labels,
               title='Parameter Combinations', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel('Log(Epoch)')
    plt.ylabel('MSE')
    plt.savefig(FIGURES_PATH + title + ' - MSE.png')
    plt.show()
    


        