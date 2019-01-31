from Cleaning import *
from News import *
from Color import *
import matplotlib.pyplot as plt
import seaborn as sns
from XML2News import *
import nltk
import sys
import time
import random
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
vectorizers = [0]*10

def preprocessing(vectorizer):

    news_list = createNews()
    list_text = []
    list_target = []
    random.shuffle(news_list)
    for news in news_list:
        news.clean_text()
        if (len(news.getCleanedText())>0):
            list_text.append(news.getCleanedText())
            list_target.append(news.getVeracity())


    appCorpus = list_text[:800]
    testCorpus = list_text[800:1100]
    appTarget = list_target[:800]
    testTarget = list_target[800:1100]
    # display(" Preprocessing : OK",'yellow')

    return process(appCorpus,appTarget,testCorpus,testTarget,vectorizer)

def process(appCorpus,appTarget,testCorpus,testTarget,vectorizer):
    joinedTestCorpus = []
    joinedAppCorpus = []
    for array in appCorpus:
        joinedAppCorpus.append(' '.join(array))
    for array in testCorpus:
        joinedTestCorpus.append(' '.join(array))
    train_vectors = vectorizer.fit_transform(joinedAppCorpus)
    # print(vectorizer.get_feature_names())
    test_vectors = vectorizer.transform(joinedTestCorpus)
    joinedAppCorpus=np.array(joinedAppCorpus)
    joinedTestCorpus=np.array(joinedTestCorpus)
    appTarget=np.array(appTarget)
    testTarget=np.array(testTarget)
    # model = DBSCAN(eps=0.3,min_samples=10).fit(train_vectors)
    model = KNeighborsClassifier(n_neighbors=2,algorithm="auto",weights='uniform').fit(train_vectors,appTarget)
    # model = MultinomialNB(alpha=0,fit_prior=False).fit(train_vectors, appTarget)

    return(model)

def eval(test):
    joinedTestCorpus = []
    joinedAppCorpus = []
    model_list = []
    list_text = []
    list_target = []
    news_list = createNews()

    for i in range(0,10):
        vectorizers[i]=TfidfVectorizer()
        model_list.append(preprocessing(vectorizers[i]))
        display("Model "+ str(i) +" : OK",'yellow')


    # Global evaluation

    if test:
        display("=>DONE Start of the evaluation ","yellow")
        random.shuffle(news_list)
        for news in news_list:
            news.clean_text()
            if (len(news.getCleanedText())>0):
                list_text.append(news.getCleanedText())
                list_target.append(news.getVeracity())


        appCorpus = list_text[:800]
        testCorpus = list_text[800:1000]
        appTarget = list_target[:800]
        testTarget = list_target[800:1000]
        for array in appCorpus:
            joinedAppCorpus.append(' '.join(array))
        for array in testCorpus:
            joinedTestCorpus.append(' '.join(array))
        #train_vectors = vectorizer.fit_transform(joinedAppCorpus)

        testTarget = np.array(testTarget)
        testTarget[testTarget=='mostly false']=int(0)
        testTarget[testTarget=='mostly true']=int(1)
        testTarget = [int(item) for item in testTarget]
        resList =[]
        for index,model in enumerate(model_list):
            test_vectors = vectorizers[index].transform(joinedTestCorpus)
            predicted = model.predict(test_vectors)
            predicted = np.array(predicted)
            predicted[predicted == 'mostly false']=0
            predicted[predicted == 'mostly true']=1
            resList.append(predicted)



        pred = [0]*len(resList[0])
        for index in range(len(resList[0])):
            temppred = 0
            for predict in resList:
                predict=np.array(predict)
                predict[predict=='mostly false']=0
                predict[predict=='mostly true']=1
                temppred+=int(predict[index])

            pred[index]=temppred
        pred = np.array(pred)/10
        pred = np.around(pred)

        display("Accuracy of the combined model = "+str(accuracy_score(testTarget,pred)),'yellow')

        confusion=confusion_matrix(testTarget, pred)

        matrice_confusion = pd.DataFrame(confusion, ["0","1"],
                      ["0","1"])
        precisionFalse = confusion[0][0]/(np.sum(confusion[0]))
        precisionTrue = confusion[1][1]/(np.sum(confusion[1]))
        display("Precision for False = "+str(precisionFalse),'yellow')
        display("Precision for True = "+str(precisionTrue),'yellow')
        pprint(matrice_confusion)
    pickle.dump(model_list,open("finalized_model.sav",'wb'))
    pickle.dump(vectorizers,open("finalized_vectorizers.sav","wb"))
    return model_list

if len(sys.argv) == 2:
    if sys.argv[1] == '-e':
        model_list = eval(True)
else:
    model_list = eval(False)
