#!/usr/bin/python3
from Cleaning import *
from News import *
import pandas as pd
from Color import *
import matplotlib.pyplot as plt
import seaborn as sns
from XML2News import *
import nltk
import time
import random
import pandas as pd
import numpy as np
# from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType,StringTensorType

vectorizer = TfidfVectorizer()

def preprocessing(appCorpus,appTarget,testCorpus,testTarget):
    """
    Function that shuffle the corpus for the differents model

    """
    # Regroupment of the 2 lists
    appBoth = list(zip(appCorpus,appTarget))
    # Shuffle
    random.shuffle(appBoth)
    # Split in two lists
    appCorpus = [x[0] for x in appBoth]
    appTest = [x[1] for x in appBoth]
    display(" Preprocessing : OK",'yellow')

    return process(appCorpus,appTarget,testCorpus,testTarget)

def process(appCorpus,appTarget,testCorpus,testTarget):
    """
    Function that train the model.
    The parameters have obvious names

    """
    # This part transform our array of array of token into array of sentence to make the Tfidf work
    joinedTestCorpus = []
    joinedAppCorpus = []
    for array in appCorpus:
        joinedAppCorpus.append(' '.join(array))
    for array in testCorpus:
        joinedTestCorpus.append(' '.join(array))
    joinedAppCorpus=np.array(joinedAppCorpus)
    joinedTestCorpus=np.array(joinedTestCorpus)
    appTarget=np.array(appTarget)
    testTarget=np.array(testTarget)

    # The vectorizer is on top of the file, it's a TfidfVectorizer without any customization
    # To ease the save of the model I used a sklearn pipeline that contain a TfidfVectorizer and a Bayésian model.
    # The bayesian model is set without any prior probability to avoid a bias due to a huge gap in the number of samples of each class
    model = MultinomialNB(alpha=0,fit_prior=False)
    pipe = make_pipeline(vectorizer,model)
    pipe.fit(joinedAppCorpus,appTarget)
    # The pipe goes into a list of model (maybe not in the future)
    return(pipe)

def eval(ev):
    joinedTestCorpus = [] # Array of sentence
    model_list = [] # List of model
    list_text = [] # temporary list of token
    list_target = [] # temporary list of veracity

    # The list is created by the file XML2News.py, it's a list of News object
    news_list = createNewsNew()
    display("=> List OK",'yellow')
    # First shuffle of the list, just to mix the false and true data from the creation.
    random.shuffle(news_list)
    # To avoid a calculation time to long I use only a part of the total list.
    news_list = news_list[:5000]
    # Treatement of the news. We only need to do it once so I don't put it in the preprocessing,
    # maybe in the future a news function to do it could be cool
    for index,news in enumerate(news_list):
        # This don't return anything, only the getters return the value
        news.clean_text()
        # I don't take the text without texts
        if (len(news.getCleanedText())>0):
            list_text.append(news.getCleanedText())
            list_target.append(news.getVeracity())
        if index%100==0 :
            print("News n°{} tagged".format(index))
    display("clean ok",'yellow')
    # This split the corpus into Learning and testing corpus with a ratio 2/3 1/3s
    # I call Corpus the text and target the veracity
    mid = 2*round((len(news_list)/3))
    appCorpus = list_text[:mid]
    testCorpus = list_text[mid:]
    appTarget = list_target[:mid]
    testTarget = list_target[mid:]

    # Here you can choose the number of model you want to train. In the eventuality of a bagging.
    nmodel =1
    for i in range(0,nmodel):

        model_list.append(preprocessing(appCorpus,appTarget,testCorpus,testTarget))
        display("Model "+ str(i) +" : OK",'yellow')
    display("=>DONE Start of the evaluation ","yellow")

    # Evaluation of the model.
    if ev:
        # Creation of the array of sentences fo the test
        for array in testCorpus:
            joinedTestCorpus.append(' '.join(array))

        testTarget = np.array(testTarget)
        testTarget[testTarget=='mostly false']=int(0)
        testTarget[testTarget=='mostly true']=int(1)
        testTarget = [int(item) for item in testTarget]
        resList =[]
        # Prediction for each model
        for index,model in enumerate(model_list):

            predicted = model.predict(joinedTestCorpus)
            predicted = np.array(predicted)
            predicted[predicted == 'mostly false']=0
            predicted[predicted == 'mostly true']=1
            predicted = [int(item) for item in predicted]
            resList.append(predicted)
            print("Model n°"+str(index)+" utilisé")
        resList = np.array(resList)
        # Vertical sum of the result.
        pred = list(map(sum,zip(*list(resList))))

        display("Accuracy of the combined model = "+str(accuracy_score(testTarget,pred)),'yellow')

        # Creation of the confusion matrix
        confusion=confusion_matrix(testTarget, pred)

        matrice_confusion = pd.DataFrame(confusion, ["0","1"],
                      ["0","1"])
        precisionFalse = confusion[0][0]/(np.sum(confusion[0]))
        precisionTrue = confusion[1][1]/(np.sum(confusion[1]))
        display("Precision for False = "+str(precisionFalse),'yellow')
        display("Precision for True = "+str(precisionTrue),'yellow')
        pprint(matrice_confusion)

        # Onnx Save (can't save a list of model for now)
        onx = convert_sklearn(model_list[0], 'Pipe',
                                     [('input', StringTensorType([1, 1]))])

        with open("Model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
    else:
        # New argument possible
    return model_list

# With or without evaluation.
if len(sys.argv) == 2:
    model_list = eval(True)
else:
    model_list = eval(False)
