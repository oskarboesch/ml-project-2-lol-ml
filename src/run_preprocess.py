from utils.utils import load_data, save_data
from preprocess.preprocess_data import preprocess_data


def main():
    """Main function to load, preprocess, and save data."""
    # Load data
    df_train, df_test, df_train_targets  = load_data()
    
    # Preprocess data
    df_train_preproc, df_train_categorical_preproc, df_test_preproc, df_train_targets_preproc  = preprocess_data(df_train,df_test, df_train_targets)

    # Save preprocessed data
    save_data(df_train_preproc, df_train_categorical_preproc, df_test_preproc,df_train_targets_preproc)

if __name__ == "__main__":
    main()