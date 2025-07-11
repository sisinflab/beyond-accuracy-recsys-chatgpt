from elliot.run import run_experiment
import pandas as pd
import os

def list_tsv_files(directory):
    # List to hold the names of tsv files
    tsv_files = []

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file is a TSV file
        if filename.endswith('.tsv'):
            tsv_files.append(filename)

    return tsv_files

def modify_tsv(input_path, output_path):
    # Read the TSV file
    df = pd.read_csv(input_path, delimiter='\t')

    # Check if the expected columns are present
    expected_columns = ['userId', 'chatGPT_title', 'dataset_name', 'itemId', 'rank']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError("Input file does not contain the expected columns.")

    # Group by userId and filter out groups with fewer than 20 items
    grouped = df.groupby('userId').filter(lambda x: len(x) >= 20)

    # Invert the order of ranks for each user based on the maximum rank value
    def invert_ranks(group):
        max_rank = group['rank'].max()
        group['rank'] = max_rank - group['rank'] + 1
        return group

    grouped = grouped.groupby('userId').apply(invert_ranks)

    # Remove specified columns
    grouped.drop(columns=['chatGPT_title', 'dataset_name'], inplace=True)

    # Save to new TSV file without header
    grouped.to_csv(output_path, sep='\t', index=False, header=False)

def clean_tsv():
    for dataset in ['facebook_book', 'hetrec2011_lastfm_2k', 'ml_small_2018']:
        directory = f"../data/dataset/{dataset}/v2/gpt-3.5-turbo-1106/ProxyDir"
        list_tsv = list_tsv_files(directory)
        for tsv in list_tsv:
            modify_tsv(f"{directory}/{tsv}", f"{directory}/clean_inverted_rank/{tsv}")

def main():
    #clean_tsv()
    #run_experiment('elliot_config_files/proxy_config_facebook.yml')
    #run_experiment('elliot_config_files/proxy_config_hetrec.yml')
    #run_experiment('elliot_config_files/proxy_config_movielens.yml')
    run_experiment('elliot_config_files/config_facebook_baseline.yml')
    run_experiment('elliot_config_files/config_hetrec_baseline.yml')
    run_experiment('elliot_config_files/config_movielens_baseline.yml.yml')


if __name__ == '__main__':
    main()
    pass