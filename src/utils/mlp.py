import torch
import torch.nn as nn
import itertools
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from pathlib import Path
import os
from matplotlib import pyplot as plt

project_root = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = project_root / 'models' / 'mlp'

# Check if folder exists
if not Path(MODEL_PATH).exists():
    Path(MODEL_PATH).mkdir(parents=True)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        current_size = input_size

        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.Dropout(p=0.5))  # Dropout with 50% probability

            layers.append(nn.ReLU())  # Using ReLU activation
            current_size = hidden_size

        # Final single output
        layers.append(nn.Linear(current_size, 1))  # Output one score per input
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

def train_mse(model, optimizer, X_train, y_train, X_val, y_val, epochs):
    # Use Mean Squared Error loss
    criterion = nn.MSELoss()
    
    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    best_mse = float('inf')
    patience = 100  # Number of epochs to wait before stopping
    epochs_no_improve = 0

    # Initialize lists to store metrics
    train_mse_list = []
    val_mse_list = []
    
    # Initialize real-time plot
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation MSE Loss")
    train_line, = ax.plot([], [], label="Train MSE", color="blue")
    val_line, = ax.plot([], [], label="Val MSE", color="orange")
    ax.legend()

    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training MSE
        model.eval()
        with torch.no_grad():
            train_predictions = model(torch.tensor(X_train, dtype=torch.float32)).squeeze()
            train_mse = criterion(train_predictions, y_train)

        # Validation
        val_predictions = model(torch.tensor(X_val, dtype=torch.float32)).detach().squeeze()
        val_mse = criterion(val_predictions, y_val)

        # Store metrics
        train_mse_list.append(train_mse)
        val_mse_list.append(val_mse)

        # Update plot
        train_line.set_xdata(range(len(train_mse_list)))
        train_line.set_ydata(train_mse_list)
        val_line.set_xdata(range(len(val_mse_list)))
        val_line.set_ydata(val_mse_list)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
                  f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")

        # Save the best model
        if val_mse < best_mse:
            best_mse = val_mse
            epochs_no_improve = 0
            save_mlp_model(model, best_mse)
            print("Model saved at", MODEL_PATH)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    plt.ioff()
    plt.show()
    plt.savefig("mlp_mse_training_plot.png")

def train_margin_ranking(model, optimizer, margin, X_train, y_train, X_val, y_val, epochs):
   

    # Use MarginRankingLoss
    criterion = nn.MarginRankingLoss(margin=margin)

    # Generate pairwise data
    X1, X2, y_pairs = generate_pairs(X_train, y_train)

    best_spearman = -float('inf')
    patience = 50  # Number of epochs to wait before stopping
    epochs_no_improve = 0

    # Initialize lists to store metrics
    train_spearman_list = []
    val_spearman_list = []

    train_mse_list = []
    val_mse_list = []

    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs1 = model(X1).squeeze()  # Predictions for first set
        outputs2 = model(X2).squeeze()  # Predictions for second set
        loss = criterion(outputs1, outputs2, y_pairs)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training Spearman
        model.eval()
        with torch.no_grad():
            train_predictions = model(X_train).squeeze()
            train_spearman, _ = spearmanr(train_predictions.numpy(), y_train.numpy())
            train_mse = mean_squared_error(train_predictions, y_train)

        # Validation
        val_predictions = model(X_val).detach().squeeze()
        val_spearman, _ = spearmanr(val_predictions, y_val)
        val_mse = mean_squared_error(val_predictions, y_val)

        # Store metrics
        train_spearman_list.append(train_spearman)
        val_spearman_list.append(val_spearman)
        train_mse_list.append(train_mse)
        val_mse_list.append(val_mse)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
                  f"Train Spearman: {train_spearman:.4f}, Val Spearman: {val_spearman:.4f}")

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            epochs_no_improve = 0
            best_model = model
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    model = best_model
            


    # Plot metrics after training
    plt.figure(figsize=(10, 6))
    plt.plot(train_spearman_list, label="Train Spearman", color="blue")
    plt.plot(val_spearman_list, label="Validation Spearman", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Spearman Correlation")
    plt.title("Training and Validation Spearman Correlation")
    plt.legend()
    plt.savefig("mlp_training_plot.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_mse_list, label="Train MSE", color="blue")
    plt.plot(val_mse_list, label="Validation MSE", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation MSE Loss")
    plt.legend()
    plt.savefig("mlp_mse_training_plot.png")
    plt.show()

    print("Training complete. Final validation Spearman:", best_spearman)

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
    pairs = list(itertools.combinations(range(len(y)), 2))  # Generate index pairs
    X1 = torch.stack([X[i] for i, j in pairs])
    X2 = torch.stack([X[j] for i, j in pairs])
    y_pairs = torch.tensor([1 if y[i] > y[j] else -1 for i, j in pairs], dtype=torch.float32)
    return X1, X2, y_pairs

def save_mlp_model(model, spearman_score):
    """
    Saves the model with the Spearman score in the file name.
    
    Parameters:
    - model: The model to save.
    - spearman_score: The Spearman correlation score (float).
    """
    score_str = f"{spearman_score:.4f}"  # Format the score to 4 digits
    score = score_str.replace('.', '_')  # Convert x_xxx to float
    model_path = os.path.join(MODEL_PATH, f"mlp_model_{score}.pth")
    torch.save(model, model_path)
    print(f"\033[92mModel saved at {model_path}\033[0m")
    return model_path

def load_mlp_model(file_name):
    model_path = os.path.join(MODEL_PATH, file_name)
    return torch.load(model_path)