import xmlschema
import os
from News import *
import json
from pprint import pprint
import csv

schema = xmlschema.XMLSchema('schema.xsd')

def getNewsFromXML(link):
    """
    Transform an article of the first news (xml) corpus into a news object.
    Parameters:
        link : Name of the file (must be in the articles folder)
    """
    # Transformation of the xml in python dictionary
    article = schema.to_dict(link)
    # creation of default values (I don't remember how to make a class with variable parameters)
    author = "Anonyme"
    mainText = "empty"
    hyperlink = []
    orientation = "default"
    # Possible value : mostly true / mixture of true and false / mostly false / no factual content
    # For the needs of the learning, I transform the 'mixture of true and false ' into 'mostly false'.
    veracity = "default"
    title = "default"

    # Recuperation of everything in the xml
    if 'author' in article:
        author = article['author']
    if 'mainText' in article:
        mainText = article['mainText']
    if 'hyperlink' in article:
        hyperlink = article['hyperlink']
    if 'orientation' in article:
        orientation = article['orientation']
    if 'veracity' in article:
        if article['veracity'] == 'mixture of true and false' :
            veracity = "mostly false"
        else:
            veracity = article['veracity']
    if 'title' in article:
        title = article['title']
    # Creation of a News instance
    newsInstance = News(author,mainText,hyperlink,orientation,veracity,title)
    return newsInstance
def createNews(method=1):
    """
    Creation of the list of News.
    Parameters:
        method: int, possible value = {1,2}. Corpus where the news are taken.
    Return:
        List of News
    """
    if method == 1:
        return getNewsFromCorpus1()
    elif method == 2:
        return getNewsFromCorpus2()

def getNewsFromCorpus1():
    """
    Creation with the First Corpus
    Return:
        List of News
    """
    news = []
    for filename in os.listdir('articles'):
        new = getNewsFromXML('articles/'+filename)
        # The News with no factual content don't have any interest in our training
        if new.getVeracity() != 'no factual content' and new.getVeracity() != 'mixture of true and false':
            news.append(new)
    return news

def getNewsFromCorpus2():
    """
    Creation with the second corpus (The big one)
    Return:
        List of News
    """
    news = []
    with open("Fake.csv","r") as f:
        # Parse of the CSV
        file = csv.reader(f, delimiter=',', quotechar='"')

        for index,line in enumerate(file):
            if index%100==0:
                print("article n°"+str(index)+ " de Fake")
            # The csv is on the shape title,text,subject,date
            text=line[1]
            title=line[0]
            article = News("author",text,"links","orientation","mostly false",title)
            news.append(article)
    # Same for the 'True' file
    with open("True.csv","r") as f:
        file = csv.reader(f, delimiter=',', quotechar='"')
        for index,line in enumerate(file):
            if index%100==0:
                print("article n°"+str(index)+ " de True")
            text=line[1]
            title=line[0]
            article = News("author",text,"links","orientation","mostly true",title)
            news.append(article)

    return news
