from Cleaning import *
from nltk.corpus import stopwords
class News(object):
    """Class that represent a News."""
    def __init__(self,author,article,links,orientation,veracity,title):
        super(News, self).__init__()
        self.hashVeracity ={"mostly true" : 4,
                            "no factual content" : 3,
                            "mixture of true and false" : 2,
                            "mostly false" : 1
                           }
        self.title=title
        self.rawArticle = str(article)
        self.author = author
        self.links = links
        self.cleanedText = []
        self.orientation = orientation
        self.veracity = veracity
    def clean_text(self):
        tokens=tokenize(self.rawArticle)
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        stemmed_token = stem(tokens)
        big_words = particle_removal(stemmed_token)
        self.cleanedText = big_words
    def getCleanedText(self):
        return self.cleanedText

    def getVeracity(self):
        return self.veracity
