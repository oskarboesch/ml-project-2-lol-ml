import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr


# Models for downstream prediction
models = {
    'Random Forest': RandomForestRegressor(),
    'MLP': MLPRegressor(),
    'Linear Regression': LinearRegression()
}

# Dataset preprocessing
non_encoded_data = np.random.rand(1000, 100)  # Example of non-encoded data
encoded_AE = np.random.rand(1000, 50)  # Example of pre-encoded data (simulating AE encoded outputs)
encoded_VAE = np.random.rand(1000, 50)  # Example of pre-encoded data (simulating VAE encoded outputs)
encoded_VAE2 = np.random.rand(1000, 50)  # Example of pre-encoded data (simulating second VAE encoded outputs)
# Example of pre-encoded data (simulating AE and VAE encoded outputs)
non_encoded_train, non_encoded_test = train_test_split(non_encoded_data, np.random.rand(1000))
encoded_AE_train, encoded_AE_test = train_test_split(encoded_AE)
encoded_VAE_train, encoded_VAE_test = train_test_split(encoded_VAE)
encoded_VAE2_train, encoded_VAE2_test = train_test_split(encoded_VAE2)

# Experiment combinations
encoded_data_sets = {
    'Non-encoded': (non_encoded_train,non_encoded_test),
    'AE': (encoded_AE_train, encoded_AE_test),
    'VAE': (encoded_VAE_train, encoded_VAE_test),
    'VAE2': (encoded_VAE2_train, encoded_VAE2_test)
}

# Target data (drug response)
y = np.random.rand(1000)
y_train, y_test = train_test_split(y)


# Cross-validation
kf = KFold(n_splits=5)
results = []

for data_name, encoded_data in encoded_data_sets.items():
    for encoder_name, encoder_fn in encoded_data_sets.items():
        
        x_train, x_test = encoder_fn
    
        for train_idx, val_idx in kf.split(x_train):
            X_fold_train, X_fold_val = x_train[train_idx], x_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            for model_name, model in models.items():
                me_scores = []
                rsq_scores = []
                spear_scores = [] 
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                me_score = mean_squared_error(y_fold_val, y_pred)
                rsq_score = r2_score(y_fold_val, y_pred)
                spear_score = spearmanr(y_fold_val, y_pred)

                me_scores.append(me_score)
                rsq_scores.append(rsq_score)
                spear_scores.append(spear_score.correlation)

            avg_me_score = np.mean(me_scores)
            avg_rsq_score = np.mean(rsq_scores)
            avg_spear_score = np.mean(spear_scores)

            results.append({
                'Encoder': encoder_name,
                'Model': model_name,
                'MSE': avg_me_score,
                'R2': avg_rsq_score,
                'Spearman': avg_spear_score
            })

# Display results
for res in results:
    print(res)
