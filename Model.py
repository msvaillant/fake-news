from Cleaning import *
from News import *
from Color import *
import matplotlib.pyplot as plt
import seaborn as sns
from XML2News import *
import nltk
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
def preprocessing():

    news_list = createNews()
    list_text = []
    list_target = []
    random.shuffle(news_list)
    for news in news_list:
        news.clean_text()
        if (len(news.getCleanedText())>0):
            list_text.append(news.getCleanedText())
            list_target.append(news.getVeracity())


    appCorpus = list_text[:1400]
    testCorpus = list_text[1400:]
    appTarget = list_target[:1400]
    testTarget = list_target[1400:]
    display(" Preprocessing : OK",'yellow')
    process(appCorpus,appTarget,testCorpus,testTarget)

def process(appCorpus,appTarget,testCorpus,testTarget):
    vectorizer = TfidfVectorizer()
    joinedAppCorpus = []
    joinedTestCorpus = []
    for array in appCorpus:
        joinedAppCorpus.append(' '.join(array))
    for array in testCorpus:
        joinedTestCorpus.append(' '.join(array))
    train_vectors = vectorizer.fit_transform(joinedAppCorpus)
    test_vectors = vectorizer.transform(joinedTestCorpus)
    joinedAppCorpus=np.array(joinedAppCorpus)
    joinedTestCorpus=np.array(joinedTestCorpus)
    appTarget=np.array(appTarget)
    testTarget=np.array(testTarget)
    print(appTarget.shape)
    print(testTarget.shape)
    model = MultinomialNB().fit(train_vectors, appTarget)
    predicted = model.predict(test_vectors)
    display("Accuracy = "+str(accuracy_score(testTarget,predicted)),'yellow')
    confusion=confusion_matrix(testTarget, predicted)
    matrice_confusion = pd.DataFrame(confusion, ["mostly false","mixture of true and false","no factual content","mostly true"],
                  ["mostly false","mixture of true and false","no factual content","mostly true"])
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(matrice_confusion, annot=True,annot_kws={"size": 16})
    plt.show()
# preprocessing()
sns.set()

x = np.linspace(0, 10, 500)
y = np.random.randn(500)
plt.plot(x,y)
plt.show()
