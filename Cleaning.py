import nltk
from nltk.stem import *
import re
from Color import *
import sys
def tokenize(text):
    list_token=nltk.word_tokenize(text)
    return list_token

def get_Article(link):
    fArticle = open(link,'r')
    rawText = fArticle.read()
    return rawText
def stem(tokens):
    list_stem = []
    stemmer = SnowballStemmer("english")
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
    try:
        text = get_Article(link)
        display("   ==> Article Retrieved",'yellow')
    except Exception as e:
        error(e)

    try:
        tokens=tokenize(text)
        display("   ==> Tokenization : OK",'yellow')
    except Exception as e:
        error(e)

    try:
        stemmed_token = stem(tokens)
        display("   ==> Stem : OK",'yellow')
    except Exception as e:
        error(e)

    try:
        big_words = particle_removal(stemmed_token)
        display("   ==> Ponctuation removal : OK",'yellow')
    except Exception as e:
        error(e)


    #print(big_words)

clean(sys.argv[1])
