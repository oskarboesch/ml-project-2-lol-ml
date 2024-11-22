from utils.get_datasets import download_datasets

if __name__ == '__main__':
    # List of gene expression datasets to download
    gene_expression_datasets = [
        ('Allen Brain Atlas Adult Human Brain Tissue Gene Expression Profiles', 'brainatlasadulthuman'),
        ('Allen Brain Atlas Adult Mouse Brain Tissue Gene Expression Profiles', 'brainatlasadultmouse'),
        ('Allen Brain Atlas Developing Human Brain Tissue Gene Expression Profiles by Microarray', 'brainatlasdevelopmentalhumanmicroarray'),
        ('Allen Brain Atlas Developing Human Brain Tissue Gene Expression Profiles by RNA-seq', 'brainatlasdevelopmentalhumanrnaseq'),
        ('Allen Brain Atlas Prenatal Human Brain Tissue Gene Expression Profiles', 'brainatlasprenatalhuman'),
        ('BioGPS Human Cell Type and Tissue Gene Expression Profiles', 'biogpshuman'),
        ('BioGPS Mouse Cell Type and Tissue Gene Expression Profiles', 'biogpsmouse'),
        ('CCLE Cell Line Gene Expression Profiles', 'cclemrna'),
        ('GDSC Cell Line Gene Expression Profiles', 'gdsc'),
        ('GTEx Tissue Gene Expression Profiles', 'gtextissue'),
        ('GTEx Tissue Sample Gene Expression Profiles', 'gtexsample'),
        ('Roadmap Epigenomics Cell and Tissue Gene Expression Profiles', 'epigenomicsmrna')
    ]

    # Download types for each dataset
    selected_downloads = [
        'gene_attribute_matrix_cleaned.txt.gz',  # Common data type for gene expression datasets
    ]

    # Call the download function with the selected gene expression datasets
    download_datasets(gene_expression_datasets, selected_downloads, decompress=True)
