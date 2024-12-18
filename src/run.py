from utils.utils import load_data, load_encoded_data, set_random_seed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils.mlp import RankMLP
import os

PREDICTION_DIR = "results/predictions/"
# Ensure the predictions directory exists
os.makedirs(PREDICTION_DIR, exist_ok=True)

RANDOM_SEED = 42
set_random_seed(RANDOM_SEED)

print("Running the pipeline...")
print("Loading data...")
# Load data
non_encoded_train, test, targets = load_data(raw=False)
_, test_raw, _ = load_data(raw=True)
cell_lines = test_raw.iloc[:, 0]  # Extract cell line identifiers

# Load encoded data
ae, _, _, pca = load_encoded_data()
nb_train_cell_lines = targets.shape[0]

# Split encoded data
encoded_data_splits = {
    "non_encoded": train_test_split(non_encoded_train, targets, test_size=0.2, random_state=RANDOM_SEED),
    "ae": train_test_split(ae[:nb_train_cell_lines], targets, test_size=0.2, random_state=RANDOM_SEED),
    "pca": train_test_split(pca[:nb_train_cell_lines], targets, test_size=0.2, random_state=RANDOM_SEED)
}

# Helper function to save predictions
def save_predictions(predictions, filename):
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, "sampleId", cell_lines)
    predictions.columns = ["sampleId", "AAC"]
    # Replace CL by TS in sampleId column
    predictions["sampleId"] = predictions["sampleId"].str.replace("CL", "TS", regex=False)
    predictions.to_csv(PREDICTION_DIR + filename, index=False)
    print(f"Predictions saved to {PREDICTION_DIR + filename}")

print("Training LR models and saving predictions...")
# Train and save Linear Regression models
for name, (X_train, X_val, y_train, y_val) in encoded_data_splits.items():
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(test if name == "non_encoded" else (ae if name == "ae" else pca)[nb_train_cell_lines:])
    save_predictions(y_pred, f"submission_lr_{name}.csv")

# Train and save RankMLP models
mlp_configs = {
    "non_encoded": {"hidden_layers": (50, 25), "learning_rate": 1e-2, "weight_decay": 1e-1},
    "ae": {"hidden_layers": (100, 50, 25), "learning_rate": 1e-2, "weight_decay": 1e-1},
    "pca": {"hidden_layers": (50, 25), "learning_rate": 1e-2, "weight_decay": 1e-1}
}

print("Training RankMLP models and saving predictions...")
for name, (X_train, X_val, y_train, y_val) in encoded_data_splits.items():
    config = mlp_configs[name]
    rankmlp = RankMLP(
        input_size=X_train.shape[1],
        hidden_layers=config["hidden_layers"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        margin=0.5,
        epochs=100,
        patience=10
    )
    print("--- Training RankMLP model on", name, "---")
    rankmlp.fit(X_train.values, y_train.values, X_val.values, y_val.values, verbose=True)
    y_pred = rankmlp.predict(test.values if name == "non_encoded" else (ae.values if name == "ae" else pca.values)[nb_train_cell_lines:])
    save_predictions(y_pred, f"submission_rankmlp_{name}.csv")

print("Pipeline completed!")
