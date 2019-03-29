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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from onnxmltools import convert_sklearn
from onnxmltools.utils import save_model
from skl2onnx.common.data_types import FloatTensorType,StringTensorType
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Model(object):
    """docstring for Model."""
    def show_most_informative_features(self,vectorizer, clf, n=20):
        feature_names = vectorizer.get_feature_names()
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            predict = self.pipe.predict([fn_2])
            if predict=='mostly false' or predict==0:
                display("\t%.4f\t%-15s\t\t%.4f\t%-15s"%(coef_1, fn_1, coef_2, fn_2),'red')
            if predict=='mostly true' or predict==1:
                display("\t%.4f\t%-15s\t\t%.4f\t%-15s"%(coef_1, fn_1, coef_2, fn_2),'yellow')
    def __init__(self,eval,save,evaldiff):
        super(Model, self).__init__()

        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB(alpha=0,fit_prior=False)
        # model = SVC(gamma=2, C=1)
        # model = MLPClassifier(alpha=1)

        # self.model = SVC(kernel="linear", C=0.025)
        self.pipe = []
        list_text = []
        list_target = []

        # The list is created by the file XML2News.py, it's a list of News object, the parameters is 1 or 2
        news_list = createNews(2)
        display("=> List OK",'yellow')
        # First shuffle of the list, just to mix the false and true data from the creation.
        random.shuffle(news_list)
        # To avoid a calculation time to long I use only a part of the total list.
        news_list = news_list[:]
        # Treatement of the news. We only need to do it once so I don't put it in the preprocessing,
        # maybe in the future a news function to do it could be cool
        for index,news in enumerate(news_list):
            # This don't return anything, only the getters return the value
            news.clean_text()
            # I don't take the text without texts
            if (len(news.getCleanedText())>0):
                list_text.append(news.getCleanedText())
                list_target.append(news.getVeracity())
            if index%1000==0 :
                print("News n{} tagged".format(index))
        display("Clean : OK",'yellow')
        # This split the corpus into Learning and testing corpus with a ratio 2/3 1/3s
        # I call Corpus the text and target the veracity
        mid = 2*round((len(news_list)/3))
        self.appCorpus = list_text[:mid]
        self.testCorpus = list_text[mid:]
        self.appTarget = list_target[:mid]
        self.testTarget = list_target[mid:]

        self.preprocessing()
        display(" Preprocessing : OK",'yellow')
        self.process()
        display(" Process : OK",'yellow')
        if eval:
            self.eval()
            display(" Evaluation : OK",'yellow')
        if save:
            self.save()
        if evaldiff:
            self.evaldiff()

    def changeModel(self,model):
        old_model = str(self.model)
        self.model = model
        display("Model change from {} to {}".format(old_model,str(model)),"yellow")
        self.preprocessing()
        self.process()
        display("News model trained","yellow")
    def preprocessing(self):
        """
        Function that shuffle the corpus for the differents model
        Shape :
            appCorpus : List of list of token
            appTarget : List of symbols {'mostly true','mostly false'}
            testCorpus : List of list of token
            testTarget : List of symbols {'mostly true','mostly false'}
        """
        # Regroupment of the 2 lists in a list of tuples
        appBoth = list(zip(self.appCorpus,self.appTarget))
        # Shuffle
        random.shuffle(appBoth)
        # Split in two lists
        self.appCorpus = [x[0] for x in appBoth]
        self.appTarget = [x[1] for x in appBoth]

    def process(self):
        """
        Function that train the model.
        The parameters have obvious names

        """
        # This part transform our array of array of token into array of sentence to make the Tfidf work
        joinedTestCorpus = []
        joinedAppCorpus = []
        for array in self.appCorpus:
            joinedAppCorpus.append(' '.join(array))
        for array in self.testCorpus:
            joinedTestCorpus.append(' '.join(array))
        joinedAppCorpus=np.array(joinedAppCorpus)
        joinedTestCorpus=np.array(joinedTestCorpus)
        self.appTarget=np.array(self.appTarget)
        print(joinedAppCorpus.shape)
        print(self.appTarget.shape)
        # The vectorizer is on top of the file, it's a TfidfVectorizer without any customization
        # To ease the save of the model I used a sklearn pipeline that contain a TfidfVectorizer and a Bayesian model.
        # The bayesian model is set without any prior probability to avoid a bias due to a huge gap in the number of samples of each class
        # model = MultinomialNB(alpha=0,fit_prior=False)
        # model = SVC(gamma=2, C=1)
        # model = MLPClassifier(alpha=1)
        self.pipe = make_pipeline(self.vectorizer,self.model)
        self.pipe.fit(joinedAppCorpus,self.appTarget)
        print(type(self.pipe.steps[1][1]))
        self.show_most_informative_features(self.vectorizer, self.pipe.steps[1][1], 20)

    def save(self):
        # Onnx Save (can't save a list of model for now)

        onx = convert_sklearn(self.pipe, 'Pipe',
                                     [('input', StringTensorType([1, 1]))])

        save_model(onx, "Model.onnx")

        print ("Model saved")
    def eval(self):
        """
        Evaluation of the model
        Parameters:
            - ev: boolean that mean evaluation or not
        """
        joinedTestCorpus = [] # Array of sentence
        model_list = [] # List of model
        list_text = [] # temporary list of token
        list_target = [] # temporary list of veracity

        # Here you can choose the number of model you want to train. In the eventuality of a bagging.
        nmodel =1
        for i in range(0,nmodel):

            model_list.append(self.pipe)
            display("Model "+ str(i) +" : OK",'yellow')
        display("=>DONE Start of the evaluation ","yellow")

        # Evaluation of the model.

        # Creation of the array of sentences fo the test
        for array in self.testCorpus:
            joinedTestCorpus.append(' '.join(array))
        self.testTarget = np.array(self.testTarget)
        self.testTarget[self.testTarget=='mostly false']=int(0)
        self.testTarget[self.testTarget=='mostly true']=int(1)
        self.testTarget = [int(item) for item in self.testTarget]
        resList =[]

        # Prediction for each model
        for index,model in enumerate(model_list):
            print(len(joinedTestCorpus))
            predicted = model.predict(np.array(joinedTestCorpus))
            predicted = np.array(predicted)
            predicted[predicted == 'mostly false']=0
            predicted[predicted == 'mostly true']=1
            predicted = [int(item) for item in predicted]
            resList.append(predicted)
            print("Model n"+str(index)+" used")
        resList = np.array(resList)
        # Vertical sum of the result.
        pred = list(map(sum,zip(*list(resList))))

        display("Accuracy of the combined model = "+str(accuracy_score(self.testTarget,pred)),'yellow')
        print(np.unique(pred))
        print(np.unique(self.testTarget))
        if len(np.unique(self.testTarget))!=2:
            return((accuracy_score(self.testTarget,pred),1,1))
        # Creation of the confusion matrix
        confusion=confusion_matrix(self.testTarget, pred)

        matrice_confusion = pd.DataFrame(confusion, ["0","1"],
                      ["0","1"])
        precisionFalse = confusion[0][0]/(np.sum(confusion[0]))
        precisionTrue = confusion[1][1]/(np.sum(confusion[1]))
        display("Precision for False = "+str(precisionFalse),'yellow')
        display("Precision for True = "+str(precisionTrue),'yellow')
        pprint(matrice_confusion)
        # return((accuracy_score(self.testTarget,pred),precisionFalse,precisionTrue))

    def evaldiff(self):
        corpus=createNews(1)
        joinedCorpus=[]
        list_text = []
        list_target = []
        for index,news in enumerate(corpus):
            # This don't return anything, only the getters return the value
            news.clean_text()
            # I don't take the text without texts
            if (len(news.getCleanedText())>0):
                list_text.append(news.getCleanedText())
                list_target.append(news.getVeracity())
            if index%1000==0 :
                print("News n{} tagged".format(index))
        display("Clean : OK",'yellow')
        for array in list_text:

            joinedCorpus.append(' '.join(array))
        corpusTarget = list_target
        print(np.unique(corpusTarget))
        print(corpusTarget)
        newcorpus = []
        for item in corpusTarget:
            if item == 'mostly false':
                newcorpus.append(0)
            else:
                newcorpus.append(1)
        corpusTarget = newcorpus

        corpusTarget = [int(item) for item in corpusTarget]
        predicted = self.pipe.predict(np.array(joinedCorpus))
        predicted[predicted == 'mostly false']=int(0)
        predicted[predicted == 'mostly true']=int(1)
        predicted = [int(item) for item in predicted]
        print(predicted)

        display("Accuracy of the combined model = "+str(accuracy_score(corpusTarget, predicted)),'yellow')
        confusion=confusion_matrix(corpusTarget, predicted)
        matrice_confusion = pd.DataFrame(confusion, ["0","1"],
                      ["0","1"])
        precisionFalse = confusion[0][0]/(np.sum(confusion[0]))
        precisionTrue = confusion[1][1]/(np.sum(confusion[1]))
        display("Precision for False = "+str(precisionFalse),'yellow')
        display("Precision for True = "+str(precisionTrue),'yellow')
        pprint(matrice_confusion)
        return((accuracy_score(corpusTarget, predicted),precisionFalse,precisionTrue))


classifiers = [MultinomialNB(alpha=0,fit_prior=False),SVC(gamma=2, C=1),MLPClassifier(alpha=1),SVC(kernel="linear", C=0.025)]
res = []
models = []
models.append(Model(False,True,False))
for model in models:
    res.append(model.eval())
    # try:
    #     res.append(model.eval())
    # except IndexError as e:
    #     pass
for line in res:

    display("Total accuracy = {}".format(line[0]),"cyan")
    display("False class accuracy = {}".format(line[1]),"magenta")
    display("True accuracy = {}".format(line[2]),"cyan")
