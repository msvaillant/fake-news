import nltk
from nltk.stem import *
import re

def tokenize(text):
    """
    Apply the nltk word_tokenize on the text
    This function call a TreeBank tokenizer
    https://www.nltk.org/_modules/nltk/tokenize/treebank.html#TreebankWordTokenizer
    """
    list_token=nltk.word_tokenize(text)
    return list_token

def get_Article(link):
    """
    Transform an filename into a text
    """
    fArticle = open(link,'r')
    rawText = fArticle.read()
    return rawText

def stem(tokens):
    """
    Use the PorterStemmer of nltk on the list of token
    https://www.nltk.org/_modules/nltk/stem/porter.html#PorterStemmer
    Return:
        List of token stemmed
    """
    list_stem = []
    stemmer = PorterStemmer()
    for token in tokens:
        list_stem.append(stemmer.stem(token))
    return list_stem
def particle_removal(tokens):
    """
    Remove the weird characters of the token list.
    """
    cleaned_list = []
    regex = re.compile('[-=@_!#$%^&*()<>?,"/\|}{~:]')
    for token in tokens:
        if (regex.search(token) == None):
            if (len(token)>2):
                cleaned_list.append(token)
    return cleaned_list
def clean(link):
    """
    Apply every operations on a filename
    """
    text = get_Article(link)
    tokens=tokenize(text)
    stemmed_token = stem(tokens)
    big_words = particle_removal(stemmed_token)

    print(big_words)
