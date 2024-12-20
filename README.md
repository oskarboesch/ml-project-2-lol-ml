# ml-project-2-lol-ml
## Drug Responses
 Project 2 from ML's course

### Overview 

The project aims to build a model to predict drug responses using cancer genomic data. 

#### Dataset 

All datasets except CGC.csv must be downloaded from kaggle at : https://www.kaggle.com/competitions/xai-in-cancer-medicine/data and placed in the data/raw folder

* train.csv: The training dataset contains rows representing different cancer cell lines and columns representing various genes. Each value corresponds to the gene expression level in a specific cancer cell line.

* train_targets.csv: This file includes the labels for the training data. The "AAC" column indicates the response of the cell lines to the drug Erlotinib, where a higher AAC value reflects a better response. Additionally, the "tissue" column identifies the type of cancer cell line. 

* test.csv: The test dataset is structured similarly to the training dataset, with rows representing cancer cell lines and columns representing gene expression features.

* CGC.csv: The CGC (Cancer Gene Census) files contains 753 unique genes linked to cancer risks and must be downloaded from the COSMIC database (https://cancer.sanger.ac.uk/cosmic) and placed in data/ folder


#### Analyses 

1. **Data exploration** (`data_exploration.ipynb`): Descriptive Statistics, Data Visualization, Missing Data Analysis, Correlation Analysis
2. **Data augmentation** (`data_augmentation.ipynb`): As our data set is small by having only 742 different cell lines and 19921 genes, we tried to add genomic data from another field collecting AAC values for similar genes in other cell lines. 
3. **Tissue analyses** (`tissue_dependent.ipynb`): We perform an analysis by tissues, by training models on only cell-line in the same tissue. We also apply a CGC dataset to filter our genes in each cell line.

### Model Architecture

Raw Data → Data Preprocessing → Feature Selection (PCA/CGC/VAE/AE) → Model Training (Linear Regression, RF, MLP) → Evaluation → Final Predictions

The model architecture involves several steps, including data preprocessing, feature selection, and the application of machine learning algorithms. We experimented with various models such as linear regression, random forest, and neural networks to identify the best-performing model for predicting drug responses.

1. **Data Preprocessing**: This step includes handling missing values and normalizing the data. (`run_preprocess.py`)
2. **Feature Selection**: We used various encoding techniques to transform the data into a more suitable format for modeling. This includes:
   - **Autoencoders (AE)**: To reduce the dimensionality of the data while preserving important features. (`auto_encoder.ipynb` + `vae.py` )
   - **Variational Autoencoders (VAE)**: To capture the underlying distribution of the data. (`auto_encoder.ipynb` + `vae.py` )
   - **Principal Component Analysis (PCA)**: To reduce the dimensionality by transforming the data into a set of orthogonal components. (`data_exploration.ipynb` + `run_preprocess.py`)
   - **Cancer Gene Census (CGC)**: To filter and select the most relevant genes for each cell line. (`run_preprocess.py`)
3. **Model Selection**: We experimented with different models including:
   - **Linear Regression**: A simple model to establish a baseline. (`evaluation.ipynb`)
   - **Random Forest**: An ensemble method that uses multiple decision trees.
   - **Neural Networks**: Specifically, Multi-Layer Perceptrons (MLP) for capturing complex patterns in the data. (`mlp.ipynb`)
4. **Final Predictions** : Final Predictions from the selected models on test.csv. (`run.py`)

### Model Performance

1. **Evaluation metric**
In this project, we primarily focus on one evaluation metric: **the Spearman score**
- Measure of the monotonic relationship between two variables
- Rank between -1 and 1 

2. **Training and Validation**: We used cross-validation to ensure the robustness of our models. The data was split into training and validation sets to evaluate the performance of the models.(`evaluation.ipynb`)

### Results

While the best Mean Squared Error (MSE) of 0.0055 (or 5.55% of the mean AAC score) was achieved with AE-RF, it fails to predict high responses to drug treatment. On the other hand, AE-RankMLP achieved a Spearman score of 0.4743 on the test dataset, indicating a weak relationship between the predicted and actual drug responses, highlighting the difficulty of the predicting task.

### Conclusion

This project demonstrates the potential of using machine learning to predict drug responses based on cancer genomic data. Future work could involve exploring additional features, improving data augmentation techniques, and experimenting with more advanced models. The results indicate that while some models perform well, there is still room for improvement, especially in terms of capturing the complex relationships in the data.

### Future Work

1. **Advanced Models**: Experiment with different encoding techniques and neural network architectures. This includes trying various autoencoders and other deep learning models to better capture the underlying patterns in the data.
2. **Feature Engineering**: Explore additional feature engineering techniques to capture more relevant information from the data.
3. **Hyperparameter Tuning**: Perform more hyperparameter tuning to optimize the performance of the models.
4. **External Data**: Incorporate external datasets to enhance the training data and improve the model's generalization capabilities.
5. **Interpretability and Explainability**: Focus on model interpretability to understand the key factors influencing drug responses. Techniques such as SHAP (SHapley Additive exPlanations) values, LIME (Local Interpretable Model-agnostic Explanations), and feature importance scores can be used to make the models more transparent and provide insights into the decision-making process. This is crucial for gaining trust in the model's predictions, especially in the medical field where understanding the reasons behind predictions is essential.

### References

1. [Cancer Gene Census (CGC)](https://cancer.sanger.ac.uk/census)
2. [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis)
3. [Spearman's Rank Correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
