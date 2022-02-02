import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer

import argparse
import configparser
import ast
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import collections

def plot_wordCloud(wordList, genre, saveloc):
    wordfreq = collections.Counter(wordList)
    # draw a Word Cloud with word frequencies
    wordcloud = WordCloud(width=900,
                          height=500,
                          max_words=200,
                          max_font_size=100,
                          relative_scaling=0.5,
                          colormap='cubehelix_r',
                          normalize_plurals=True).generate_from_frequencies(wordfreq)
    plt.figure(figsize=(17, 14))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word cloud for Genre '+genre)
    plt.axis("off")
    plt.savefig(saveloc)

def plotGenreGraph(df, png_loc):
    g = df.nlargest(columns="Count", n=30)
    plt.figure(figsize=(12, 15))
    ax = sns.barplot(data=g, x="Count", y="Genre")
    ax.set(ylabel='Count')
    plt.savefig(png_loc)


def visualizeGenreData(genres, png_loc):
    genres_list = list()
    for i in genres:
        genres_list.append(ast.literal_eval(i))
    all_genres = sum(genres_list, [])
    all_genres = nltk.FreqDist(all_genres)
    all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()),'Count': list(all_genres.values())})
    plotGenreGraph(all_genres_df, png_loc)
    return


def freq_words(x, png_file, terms=30):
    all_words = ' '.join([text for text in str(x).split()])
    all_words = all_words.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})
    d = words_df.nlargest(columns="count", n=terms)

    # visualize words and frequencies
    plt.figure(figsize=(12, 15))
    ax = sns.barplot(data=d, x="count", y="word")
    ax.set(ylabel='Word')
    plt.savefig(png_file)

def get_words_for_genre(genreList, df):
    map = dict()
    for genre in genreList:
        words = list()
        for index, row in df.iterrows():
            if(genre in row['Genres'].split(',')):
                words.append(str(row['cleaned_text']).split())
        map[genre] = words

    dup_map = dict()
    for key in map:
        wordsList = set()
        for words in map[key]:
            for word in words:
                wordsList.add(word)
        dup_map[key] = wordsList
    return dup_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    combined_dataset = config.get('04_exploratory_data_analysis', 'combined_dataset')
    genre_visualization_loc = config.get('04_exploratory_data_analysis', 'genre_count_visualization')
    word_freq_visualization_loc = config.get('04_exploratory_data_analysis', 'word_freq_visualization')
    stopword_removal_visualization_loc = config.get('04_exploratory_data_analysis', 'stopword_removal_visualization')
    if not os.path.isfile(combined_dataset):
        print('Please provide a valid location of the final dataset.')
        exit()

    cleaned_final_dataset = config.get('04_exploratory_data_analysis', 'cleaned_final_dataset')
    if not os.path.isfile(cleaned_final_dataset):
        print('Please provide a valid location of the cleaned dataset.')
        exit()

    df = pd.read_csv(combined_dataset, error_bad_lines=False)
    visualizeGenreData(df['Genres'], genre_visualization_loc)
    df['text'] = df['text'].astype('category')
    freq_words(df['text'], word_freq_visualization_loc, 50)
    #
    df_cleaned = pd.read_csv(cleaned_final_dataset, error_bad_lines=False)
    freq_words(df_cleaned['cleaned_text'], stopword_removal_visualization_loc, 50)
    df_cleaned['Genres'] = df_cleaned['Genres'].apply(lambda x: ast.literal_eval(x))
    df_cleaned['Genres'] = df_cleaned['Genres'].apply(lambda x: ','.join(x))
    genreList = ['mystery', 'religious', 'cookery', 'fiction', 'children', 'war', 'history', 'thriller', 'literature', 'music', 'utopian', 'satire', 'evil', 'environmental', 'comic', 'horror', 'horrow', 'documentary', 'adventure', 'erotica', 'fantasy', 'business', 'romance', 'educational', 'action', 'comedy', 'anthropology', 'philosophy', 'nonfiction', 'biography', 'sorrow']
    map = get_words_for_genre(genreList, df_cleaned)

    saveLoc = '../../visualization_data/wordcloud/wordCloud_'
    for key in map:
        loc_final = saveLoc+key+'.png'
        if len(map[key])>0:
            plot_wordCloud(list(map[key]), key, loc_final)
    print(map)
