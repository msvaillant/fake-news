import nltk
from nltk.stem import *
import re

def tokenize(text):
    list_token=nltk.word_tokenize(text)
    return list_token

def get_Article(link):
    fArticle = open(link,'r')
    rawText = fArticle.read()
    return rawText
def stem(tokens):
    list_stem = []
    stemmer = PorterStemmer()
    for token in tokens:
        list_stem.append(stemmer.stem(token))
    return list_stem
def particle_removal(tokens):
    cleaned_list = []
    regex = re.compile('[-=@_!#$%^&*()<>?,"/\|}{~:]')
    for token in tokens:
        if (regex.search(token) == None):
            if (len(token)>2):
                cleaned_list.append(token)
    return cleaned_list
def clean(link):
    text = get_Article(link)
    tokens=tokenize(text)
    stemmed_token = stem(tokens)
    big_words = particle_removal(stemmed_token)

    print(big_words)
