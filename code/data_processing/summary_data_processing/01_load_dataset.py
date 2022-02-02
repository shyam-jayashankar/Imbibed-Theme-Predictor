import argparse
import configparser
import json
import os.path

import pandas as pd

def correctGenresInDataset(dataset):
    genres = []
    for i in dataset['Genres']:
        genres.append(list(json.loads(i).values()))

    dataset.drop('Genres', axis = 1, inplace=True)
    dataset['Genres'] = genres
    return dataset

def readDataset(fileName):
    dataframe = pd.read_csv(fileName, error_bad_lines=False, encoding='unicode_escape')
    dataframe = dataframe[dataframe['Genres'].notna()]
    dataframe = dataframe[['Genres', 'Plot_summary']]
    return dataframe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    train_data_loc = config.get('01_load_summary_dataset', 'summary_dataset_loc')
    if not os.path.isfile(train_data_loc):
        print('Please provide a valid location of the dataset.')
        exit()

    df = readDataset(train_data_loc)
    df = correctGenresInDataset(df)

    log_file = config.get('01_load_summary_dataset', 'processed_summary_dataset_loc')
    df.to_csv(log_file, index=False)
    print('Saved preprocessed result to log file!')





