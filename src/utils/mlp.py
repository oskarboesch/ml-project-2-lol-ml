import torch
import torch.nn as nn
import itertools
from scipy.stats import spearmanr


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
def train_margin_ranking(model, optimizer, margin, X_train, y_train, X_val, y_val, epochs):
    # Use MarginRankingLoss
    criterion = nn.MarginRankingLoss(margin=margin)

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Generate pairwise data
    X1, X2, y_pairs = generate_pairs(X_train, y_train)

    best_spearman = -float('inf')
    patience = 100  # Number of epochs to wait before stopping
    epochs_no_improve = 0

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

        # Validation
        model.eval()
        val_predictions = model(torch.tensor(X_val, dtype=torch.float32)).detach().squeeze()
        val_spearman, _ = spearmanr(val_predictions, y_val)

        # Evaluate Spearman correlation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(torch.tensor(X_train, dtype=torch.float32)).squeeze()
                spearman, _ = spearmanr(predictions.numpy(), y_train.numpy())
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Spearman: {spearman:.4f}", "Validation Spearman:", val_spearman)
        
        if val_spearman > best_spearman:
            best_spearman = val_spearman
            epochs_no_improve = 0
            best_model = model.state_dict()  # Save the best model
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            model.load_state_dict(best_model)  # Load the best model
            break
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