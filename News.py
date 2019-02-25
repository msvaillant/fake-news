from Cleaning import *
import numpy as np
from nltk.corpus import stopwords
from nltk import pos_tag,word_tokenize
class News(object):
    """Class that represent a News."""
    def __init__(self,author,article,links,orientation,veracity,title):
        super(News, self).__init__()
        self.hashVeracity ={"mostly true" : 4,
                            "no factual content" : 3,
                            "mixture of true and false" : 2,
                            "mostly false" : 1
                           }
        self.hashGrammar ={ "CC" : 33,
                            "CD" : 1,
                            "DT" : 2,
                            "EX" : 3,
                            "IN" : 4,
                            "JJ" : 5,
                            "JJR": 6,
                            "JJS": 7,
                            "LS" : 8,
                            "MD" : 9,
                            "NN" : 10,
                            "NNP": 11,
                            "NNS": 12,
                            "PDT": 13,
                            "POS": 14,
                            "PRP": 15,
                            "PRP$":16,
                            "RB" : 17,
                            "RBR": 18,
                            "RBS": 19,
                            "RP" : 20,
                            "TO" : 21,
                            "UH" : 22,
                            "VB" : 23,
                            "VBD": 24,
                            "VBG": 25,
                            "VBN": 26,
                            "VBP": 27,
                            "VBZ": 28,
                            "WDT": 29,
                            "WP" : 30,
                            "WRB": 31,
                            "UNK": 32
                            }
        self.title=str(title)
        self.rawArticle = str(article)
        self.author = author
        self.links = links
        self.cleanedText = []
        self.orientation = orientation
        self.veracity = veracity
        self.taggedTextTemp=[]
        self.taggedText=[]
    def clean_text(self):
        tokens=tokenize(self.title)
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        stemmed_token = stem(tokens)
        big_words = particle_removal(stemmed_token)
        self.cleanedText = big_words
        
    def tag_text(self):
        for sen in self.rawArticle.split('.'):
            self.taggedTextTemp.append(nltk.pos_tag(word_tokenize(sen)))

    def getTaggedText(self):
        for sentences in self.taggedTextTemp:
            sent = []

            for word in sentences:
                try:
                    sent.append(self.hashGrammar[word[1]])
                except KeyError as e:
                    sent.append(32)
            padmax=300
            if len(sent)<=padmax:
                self.taggedText.append(sent+[0]*(padmax-len(sent)))
            else:
                self.taggedText.append(sent[:padmax])

        return self.taggedText
    def getCleanedText(self):
        return self.cleanedText

    def getVeracity(self):
        return self.veracity
