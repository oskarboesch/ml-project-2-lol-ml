from utils.utils import load_data, save_data
from preprocess.preprocess_data import preprocess_data, create_CGC_data


def main():
    """Main function to load, preprocess, and save data."""
    # Load data
    df_train, df_test, df_train_targets  = load_data()
    
    # Preprocess data
    df_train_preproc, df_train_categorical_preproc, df_test_preproc, df_train_targets_preproc, df_total  = preprocess_data(df_train,df_test, df_train_targets)

    # Add CGC Data
    df_train_cgc, df_test_cgc = create_CGC_data(df_train_preproc, df_test_preproc)

    # Save preprocessed data
    save_data(df_train_preproc, df_train_categorical_preproc, df_test_preproc,df_train_targets_preproc, df_total, df_train_cgc, df_test_cgc)

if __name__ == "__main__":
    main()