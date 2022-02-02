import argparse
import configparser
import os

import pandas as pd


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    #absolute path of the file folder is required.
    path = config.get('02_merge_books_to_df', 'books_chapter_loc')

    map = dict()
    for i in range(1,8):
        p = path + str(i)
        os.chdir(p)
        for file in os.listdir():
            if file.endswith(".txt"):
                file_path = f"{p}/{file}"
                map[str(i)+file] = read_text_file(file_path)


    df_trained = pd.DataFrame(list(map.items()), columns=['fileName', 'chapter'])
    print(df_trained.head())
    print(df_trained.shape)

    log_file =  config.get('02_merge_books_to_df', 'processed_books_dataset_loc')
    df_trained.to_csv(log_file)
    print('trained data saved!')

    map = dict()
    for i in range(8, 12):
        p = path + str(i)
        os.chdir(p)
        for file in os.listdir():
            if file.endswith(".txt"):
                file_path = f"{p}/{file}"
                map[str(i) + file] = read_text_file(file_path)

    df_test = pd.DataFrame(list(map.items()), columns=['fileName', 'chapter'])
    print(df_test.head())
    print(df_test.shape)

    log_file = config.get('02_merge_books_to_df', 'test_books_dataset_loc')
    df_test.to_csv(log_file)

    print('test data saved!')





