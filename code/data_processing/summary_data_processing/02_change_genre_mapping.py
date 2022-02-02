import argparse
import ast
import configparser
import csv
import os.path

import pandas as pd

complete_set = set()

def read_csv_to_dict(file_loc):
    with open(file_loc, mode='r', encoding='utf8') as inp:
        reader = csv.reader(inp)
        dict_from_csv = {rows[0].lower(): rows[1].lower().split(',') for rows in reader}

    return dict_from_csv

def neutralize_genre(genres, genre_map):
    genre_list = set()
    for genre in genres.split(','):
        if genre.lower() in genre_map:
            for i in genre_map[genre.lower()]:
                if i != "":
                    genre_list.add(i.replace(" ",""))
                    complete_set.add(i.replace(" ",""))
    return list(genre_list)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    train_data_loc = config.get('01_load_summary_dataset', 'processed_summary_dataset_loc')
    if not os.path.isfile(train_data_loc):
        print('Please provide a valid location of the dataset.')
        exit()

    genre_mapping_loc = config.get('01_change_genre_mapping', 'genre_mapping_file')
    if not os.path.isfile(train_data_loc):
        print('Please provide a valid location of the map.')
        exit()
    df = pd.read_csv(train_data_loc, error_bad_lines=False)
    genre_map = read_csv_to_dict(genre_mapping_loc)

    df['Genres'] = df['Genres'].apply(lambda x: ast.literal_eval(x))
    df['Genres'] = df['Genres'].apply(lambda x: ','.join(x))
    df['Genres'] = df['Genres'].apply(lambda x: neutralize_genre(x,genre_map))
    df['text'] = df['Plot_summary']

    df_final = df[['Genres', 'text']]
    print(list(complete_set))
    print(len(complete_set))

    log_file = config.get('01_change_genre_mapping', 'genre_mapped_summary_dataset_loc')
    df_final.to_csv(log_file, index=False)


