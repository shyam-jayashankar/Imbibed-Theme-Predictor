import string
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sqlite3
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
from sqlalchemy import create_engine
import datetime as dt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from skmultilearn.adapt import mlknn
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle

import glob
import argparse
import configparser
import ast
from collections import Counter
from sklearn.pipeline import make_pipeline,FeatureUnion,Pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    train_data_loc = config.get('05_one_vs_rest_model', 'train_data_loc')
    if not os.path.isfile(train_data_loc):
        print('Please provide a valid location of the dataset.')
        exit()

    df = pd.read_csv(train_data_loc, error_bad_lines=False)
    df['Genres'] = df['Genres'].apply(lambda x: ast.literal_eval(x))
    df['Genres'] = df['Genres'].apply(lambda x: ','.join(x))

    df.dropna(inplace=True)

    X = df['cleaned_text']
    y = df['Genres']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    vectorizer = CountVectorizer(tokenizer= lambda x: x.split(','), binary='true',max_features=3)
    y_train_vect = vectorizer.fit_transform(y_train)
    y_test_vect = vectorizer.transform(y_test)

    start = datetime.now()
    vectorizer_tfidf = TfidfVectorizer(min_df=10, max_features=20000, smooth_idf=True, norm="l2", \
                                 tokenizer= lambda x:x.split(), sublinear_tf=False, ngram_range=(1,4))
    x_train_ngram = vectorizer_tfidf.fit_transform(X_train)
    x_test_ngram = vectorizer_tfidf.transform(X_test)
    print("Time taken to vectorize train data: ",datetime.now() - start)

    classifier = OneVsRestClassifier(SGDClassifier(loss='log', penalty='l2', class_weight='balanced'))
    parameter = {"estimator__alpha":[10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0]}
    gsv = GridSearchCV(classifier, param_grid=parameter, scoring='f1_micro', n_jobs=-1)

    gsv.fit(x_train_ngram, y_train_vect)

    print(gsv.best_score_)
    print(gsv.best_params_)
    #
    # classifier_final = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.0001, penalty='l2', class_weight='balanced'), n_jobs=-1)
    # classifier_final.fit(x_train_ngram, y_train_vect)

    model_loc = config.get('05_one_vs_rest_model', 'model_loc')
    # with open(model_loc, 'wb') as handle:
    #     pickle.dump(classifier_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(model_loc, 'rb') as handle:
        model = pickle.load(handle)
    y_pred = model.predict(x_test_ngram)

    print("Accuracy :", metrics.accuracy_score(y_test_vect, y_pred))
    print("Hamming_loss :", metrics.hamming_loss(y_test_vect, y_pred))
    print()
    precision = precision_score(y_test_vect, y_pred, average='micro')
    recall = recall_score(y_test_vect, y_pred, average='micro')
    f1 = f1_score(y_test_vect, y_pred, average='micro')

    print("Micro-averaged quality scoring")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    print()
    precision = precision_score(y_test_vect, y_pred, average='macro')
    recall = recall_score(y_test_vect, y_pred, average='macro')
    f1 = f1_score(y_test_vect, y_pred, average='macro')

    print("macro-averaged quality scoring")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))






