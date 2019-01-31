import xmlschema
import os
from News import *
import json
from pprint import pprint
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
        if article['veracity'] == 'mixture of true and false':
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
