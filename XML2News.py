import xmlschema
import os
from News import *
import json
from pprint import pprint
import csv

schema = xmlschema.XMLSchema('schema.xsd')

def getNewsFromXML(link):
    article = schema.to_dict(link)

    author = "Anonyme"
    mainText = "empty"
    hyperlink = []
    orientation = "default"
    #Mostly True / mixture of True and false / Mostly false / No factual content
    veracity = "default"
    title = "default"
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
    newsInstance = News(author,mainText,hyperlink,orientation,veracity,title)
    return newsInstance
def createNews():
    news = []
    for filename in os.listdir('articles'):
        new = getNewsFromXML('articles/'+filename)
        if new.getVeracity() != 'no factual content' and new.getVeracity() != 'mixture of true and false':
            news.append(new)
        #news.append(json.dumps(getNewsFromXML('articles/'+filename).__dict__))
    return news
def createNewsNew():
    news = []
    with open("Fake.csv","r") as f:
        file = csv.reader(f, delimiter=',', quotechar='"')
        for index,line in enumerate(file):
            print("article n°"+str(index)+ " de Fake")
            text=line[1]
            title=line[0]
            article = News("author",text,"links","orientation","mostly false",title)
            news.append(article)
    with open("True.csv","r") as f:
        file = csv.reader(f, delimiter=',', quotechar='"')
        for index,line in enumerate(file):
            print("article n°"+str(index)+ " de True")
            text=line[1]
            title=line[0]
            article = News("author",text,"links","orientation","mostly true",title)
            news.append(article)

    return news
