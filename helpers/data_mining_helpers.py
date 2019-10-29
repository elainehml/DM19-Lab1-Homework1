import nltk
import math
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob



"""
Helper functions for data mining lab session 2018 Fall Semester
Author: Elvis Saravia
Email: ellfae@gmail.com
"""

def format_rows(docs):
    """ format the text field and strip special characters """
    D = []
    for d in docs.data:
        temp_d = " ".join(d.split("\n")).strip('\n\t')
        D.append([temp_d])
    return D

def format_labels(target, docs):
    """ format the labels """
    return docs.target_names[target]

def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter+=1
    return ("The amoung of missing records is: ", counter)

def tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using the nltk library
    """
    tokens = []
    for d in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(d, language='english'):
            # filters here
            tokens.append(word)
    return tokens

def lowercase_nopunc(text):
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    dealed = lowers.translate(remove_punctuation_map)
    return dealed

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def lemma_tokens(tokens,lemmatizer):
    lemmaed = []
    for item in tokens:
        lemmaed.append(lemmatizer.lemmatize(item))
    return lemmaed

def postagging(text):
    result = TextBlob(text)
    return result.tags

def stop_remove(tokens):
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    return filtered

def TFs(text):
    dealed = lowercase_nopunc(text)
    tokens = tokenize_text(dealed)
    filtered = stop_remove(tokens)
    stemmer = PorterStemmer()
    stemmed = stem_tokens(filtered, stemmer)
    count = Counter(stemmed)
    print (count.most_common(20))
    freq_df=pd.DataFrame.from_records(count.most_common(20),columns=['token','count'])
    freq_df.plot(kind='bar',x='token');
    
def TFl(text):
    dealed = lowercase_nopunc(text)
    tokens = tokenize_text(dealed)
    filtered = stop_remove(tokens)
    lemmatizer=WordNetLemmatizer()
    lemmaed = lemma_tokens(filtered, lemmatizer)
    count = Counter(lemmaed)
    print (count.most_common(20))
    freq_df=pd.DataFrame.from_records(count.most_common(20),columns=['token','count'])
    freq_df.plot(kind='bar',x='token');

def wf(word, count_list):
    return sum(count[word] for count in count_list if word in count)

def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)

def ilf(word, count_list):
    return int(len(count_list) / (n_containing(word, count_list)))

def wfilf(word, count, count_list):
    return wf(word, countlist) * ilf(word, count_list)

def sort_by_value(d):
    
    items=d.items()

    backitems=[[v[1],v[0]] for v in items]

    backitems.sort(reverse=True)

    return [ backitems[i][1] for i in range(0,len(backitems))]



