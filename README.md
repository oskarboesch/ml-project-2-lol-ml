# ml-project-2-lol-ml
## Drug Responses
 Project 2 from ML's course

### Overview 

The project aims to build a model to predict durg responses using cancer genomic data. 

#### Dataset 

* train.csv: The training dataset contains rows representing different cancer cell line and columns representing various genes. Each value corresponds to the gene expression level in a specific cancer cell line.

* train_targets.csv: This file includes the labels for the training data. The "AAC" column indicates the response of the cell lines to the drug Erlotinib, where a higher AAC value reflects a better response. Additionally, the "tissue" column identifies the type of cancer cell line. 

* test.csv: The test dataset is structured similarly to the training dataset, with rows representing cancer cell lines and columns representing gene expression features. 

#### Analyses 

1. **Data exploration** (notebook: data_exploration): Descriptive Statistics, Data Visualization, Missing Data Analysis, Correlation Analysis
2. **Data augmentation**(notebook: data_augementation): As our data set is small by having only 742 different cell lines and 19921 genes, we tried to add genomic data from another field collecting AAC values for similar genes in other cell lines. 
3. **Tissue_analyses** (notebook:tissue_analysis): We perform an analysis by tissues, by training models on only cell-line in the same tissue. We also apply a CGC dataset to filter our genes in each cell line. 

### Model Architecture


### Model Architecture

### Model Performance: 
1. **Evaluation metric**
In this project, we look at mainly one evalation metric: **the spearman score**
- measure of the monotonic relationship between two variables
- Ranke between -1 and 1 

