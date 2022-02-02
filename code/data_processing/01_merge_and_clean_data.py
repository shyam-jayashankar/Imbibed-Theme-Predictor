import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import re
import os
import itertools
import ast
import argparse
import configparser

contractions = {"ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not",
                "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not","haven't": "have not", "he'd": "he would",  "he'd've": "he would have", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would",
                "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "must've": "must have",
                "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not", "shan't": "shall not", "sha'n't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not", "that'd": "that would",
                "that's": "that is", "there'd": "there had", "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
                "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "where'd": "where did", "where's": "where is","who'll": "who will", "who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                "you'll": "you will",   "you're": "you are",   "thx" : "thanks"}

myOwnStopList=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'what','how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'do', 'should', "should", 'now', 'd', 'm', 'o', 're', 've', 'y', 'ain', "are", 'could', "was",
'would','have','get','got','getting','one','two','still','going']

def remove_contractions(text):
    return contractions[text.lower()] if text.lower() in contractions.keys() else text

# clean dataset
def clean_dataset(text):
    text = text.lower()
    text = re.sub(r'#','', text)
    text = re.sub(r'\&\w*;', '', text)
    text = re.sub(r'\$\w*', '', text)
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)
    text = re.sub(r'\s\s+','', text)
    text = re.sub(r'[ ]{2, }',' ',text)
    text=  re.sub(r'http(\S)+', '',text)
    text=  re.sub(r'http ...', '',text)
    text=  re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+','',text)
    text=  re.sub(r'RT[ ]?@','',text)
    text = re.sub(r'@[\S]+','',text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'&amp;?', 'and',text)
    text = re.sub(r'&lt;','<',text)
    text = re.sub(r'&gt;','>',text)
    text = re.sub("[^a-zA-Z]", ' ', text)
    text = re.sub(r'([\w\d]+)([^\w\d ]+)', '\1 \2',text)
    text = re.sub(r'([^\w\d ]+)([\w\d]+)', '\1 \2',text)
    text= ''.join(c for c in text if c <= '\uFFFF')
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub("\'", "", text)
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    text = ' '.join(re.sub("[\.\,\!\?\:\;\-\=\/\|\'\(\']", " ", text).split())
    text = text.replace(":"," ")
    text = re.sub("([^\x00-\x7F])+"," ",text)
    text = ' '.join(re.sub("[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
    text = ' '.join(text.split())
    text = text.lower()
    return text

def clean_chapter(df):
    df['cleaned_text'] = df['text'].apply(lambda x: remove_contractions(str(x)))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: clean_dataset(x))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (myOwnStopList)]))
    return df[['Genres','cleaned_text']]

def clean_testData(df):
    df['cleaned_text'] = df['chapter'].apply(lambda x: remove_contractions(str(x)))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: clean_dataset(x))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (myOwnStopList)]))
    return df[['cleaned_text']]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))

    summary_data = config.get('03_merge_and_clean_df', 'summary_processed_data')
    if not os.path.isfile(summary_data):
        print('Please provide a valid location of the dataset.')
        exit()
    summary_data_df = pd.read_csv(summary_data, error_bad_lines=False)

    books_data = config.get('03_merge_and_clean_df', 'books_processed_data')
    if not os.path.isfile(books_data):
        print('Please provide a valid location of the dataset.')
        exit()
    books_data_df = pd.read_csv(books_data, error_bad_lines=False)

    merged_final_df = summary_data_df.append(books_data_df)
    merged_final_df.drop(['Unnamed: 0'], inplace=True, axis=1)
    merged_final_df['text'] = merged_final_df['text'].astype('category')
    merged_final_df['Genres'] = merged_final_df['Genres'].astype('category')

    final_df_loc = config.get('03_merge_and_clean_df', 'combined_dataset')
    merged_final_df.to_csv(final_df_loc)

    print('Starting to clean training Dataset data')

    df_final = clean_chapter(merged_final_df)
    df_final['cleaned_text'] = df_final['cleaned_text'].astype('category')
    df_final['Genres'] = df_final['Genres'].astype('category')

    books_training_data_cleaned = config.get('03_merge_and_clean_df', 'cleaned_final_dataset')
    df_final.to_csv(books_training_data_cleaned)
    print('Cleaned training dataset!')

    print('Reading test dataset for data cleaning')
    test_data_loc = config.get('03_merge_and_clean_df', 'final_test_dataset')
    df_test = pd.read_csv(test_data_loc, error_bad_lines=False)
    print('Starting to clean test Dataset data')

    df_final_test = clean_testData(df_test)
    print(df_final.dtypes)

    books_test_data_cleaned = config.get('03_merge_and_clean_df', 'cleaned_test_dataset')
    df_final_test.to_csv(books_test_data_cleaned)
    print('Cleaned test dataset!')


