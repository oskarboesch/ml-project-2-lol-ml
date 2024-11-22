import requests
import os
import zlib

# Ensure the extra_data folder exists
extra_data_folder = 'data/extra_data'
if not os.path.exists(extra_data_folder):
    os.makedirs(extra_data_folder)

def _download_file(response, filename):
    # Construct the full path inside the extra_data folder
    file_path = os.path.join(extra_data_folder, filename)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

def _download_and_decompress_file(response, filename):
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    filename = filename[:-3]  # Remove the '.gz' extension
    
    with open(filename, 'w+') as f:
        while True:
            chunk = response.raw.read(1024)
            if not chunk:
                break
            string = decompressor.decompress(chunk)
            f.write(string.decode('utf-8'))

def download_datasets(selected_datasets, selected_downloads, decompress=False):
    for dataset, path in selected_datasets:
        dataset_folder = os.path.join(extra_data_folder, dataset)  # Define dataset folder in extra_data
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        for downloadable in selected_downloads:
            url = f'https://maayanlab.cloud/static/hdfs/harmonizome/data/{path}/{downloadable}'
            response = requests.get(url, stream=True)
            filename = os.path.join(dataset_folder, downloadable)  # Construct the full path for the downloadable file

            # Not every dataset has all downloadables.
            if response.status_code != 200:
                continue

            if decompress and 'txt.gz' in filename:
                _download_and_decompress_file(response, filename)
            else:
                _download_file(response, filename)

        print(f'{dataset} downloaded.')

