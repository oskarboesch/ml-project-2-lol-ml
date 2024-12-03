from utils.utils import load_data, save_data, save_to_tsv
from preprocess.preprocess_data import preprocess_data, create_CGC_data, create_PCA_data


def main():
    """Main function to load, preprocess, and save data."""
    # Load data
    df_train, df_test, df_train_targets  = load_data()
    
    # Preprocess data
    df_train_preproc, df_train_categorical_preproc, df_test_preproc, df_train_targets_preproc, df_total  = preprocess_data(df_train,df_test, df_train_targets)

    # MIn-Max Scale for VAE2
    df_train_min_max, _, df_test_min_max, _, df_total_min_max  = preprocess_data(df_train,df_test, df_train_targets, min_max_scale=True)


    # Add CGC Data
    df_train_cgc, df_test_cgc = create_CGC_data(df_train_preproc, df_test_preproc)

    # Add PCA Data
    df_train_pca, df_test_pca = create_PCA_data(df_train_preproc, df_test_preproc)

    # Add CGC Data for Min-Max
    df_train_cgc_min_max, df_test_cgc_min_max = create_CGC_data(df_train_min_max, df_test_min_max)

    # Add PCA Data for Min-Max
    df_train_pca_min_max, df_test_pca_min_max = create_PCA_data(df_train_min_max, df_test_min_max)

    # Save preprocessed data
    save_data(df_train_preproc, df_train_categorical_preproc, df_test_preproc,df_train_targets_preproc, df_total, df_train_cgc, df_test_cgc, df_train_pca, df_test_pca)
    save_to_tsv(df_total_min_max, df_train_cgc_min_max, df_test_cgc_min_max, df_train_pca_min_max, df_test_pca_min_max)

if __name__ == "__main__":
    main()