# chapter-genre-prediction
Predicting Genres for chapters of a book. 

Project built on **python 3.8**

The data collection, preprocessing and EDA of the data was done using python

Developed 3 models:
1. **Model 1**: Tf-idf + OneVsRest classifier
2. **Model 2:** Tf-idf + Classifier Chains with binary relevance
3. **Model 3:** Sentence Embedding + Classifier Chains with binary relevance

* The code can be found in _/code_ directory.
  * Data Processing is found in _/code/data_processing_ directory.
    * _/book_processing_ -  code for chunking books and cleaning them.
    * _/summary_data_processing_ - code for cleaning the CMU dataset can be found here
    * _01_merge_and_clean_data.py_ - Merges both books and summary data into a single dataset and does data cleaning
    * _02_exploratory_data_analysis.py_ - Exploratory data analysis of the merged data set is being performed here
  * BaseLine Model - For the baseline model, we used the 'OneVsRest' classifier can be found in _/code/model1-OnevsRest_.
  * **config.ini** - contains all the configurations for the .py files.

* The dataset can be found in _/dataset_ directory. 
  * _/raw_data_ - contains all the books in txt format. eg: book1.txt etc.
  * _/raw_data/books/book*_ - contains the chapter wise segregation for each book.
  * _/raw_data/dataset.csv_ - CMU summary dataset.
  * _/preprocessed_ - contains all the preprocessed data.

* The trained models are saved accordingly in _/models_ directory.
  * _/OneVsRest_ - contains .pkl file for OneVsRest classifier.
  * _/binary_relevance_models_ - contains .sav files saved for each timestep of binary relevance classifier trained with sentence embeddings.
  * _/classifier_chain_models_ - contains .sav files saved for each timestep of classifier chains models trained with sentence embeddings.
  * _/classifier_models_ - contains .sav files for tf-idf for classifier chain model

* The EDA results are stored in _/visualization_data_ directory.
* Code for Binary relevance classifier and classifier chain model can be found in _/jupyter_notebook_.
  * The dataset used for notebooks also can be found separately in _/jupyter_notebook/dataset_. 
  * _dsf_multilabel_classification.ipynb_ - Model 3 implementation
  * _TF-idf dsf_multilabel_classification.ipynb_ - Model 2 implementation
  * _dsf_unsupervised_annotation.ipynb_ - Unsupervised learning based on centroid method for predicting genres of a book.


* Running _/code/data_processing/books_processing/01_chapterize_books.py_
```buildoutcfg
python /code/data_processing/book_processing/01_chapterize_books.py /dataset/raw_data/books/book1.txt
```
- Running the other files
```buildoutcfg
python <filename.py> --config_file config.ini
```
eg:
```buildoutcfg
python code/data_processing/02_exploratory_data_analysis.py --config_file ../config.ini
```




