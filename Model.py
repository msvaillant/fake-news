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
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import TensorType
vectorizer = TfidfVectorizer()
def preprocessing(appCorpus,appTarget,testCorpus,testTarget):


    appBoth = list(zip(appCorpus,appTarget))
    random.shuffle(appBoth)
    appCorpus = [x[0] for x in appBoth]
    appTest = [x[1] for x in appBoth]
    display(" Preprocessing : OK",'yellow')

    return process(appCorpus,appTarget,testCorpus,testTarget)

def process(appCorpus,appTarget,testCorpus,testTarget):
    joinedTestCorpus = []
    joinedAppCorpus = []
    for array in appCorpus:
        joinedAppCorpus.append(' '.join(array))
    for array in testCorpus:
        joinedTestCorpus.append(' '.join(array))
    train_vectors = vectorizer.fit_transform(joinedAppCorpus)
    # train_vectorst = np.array(appCorpus)
    # print(train_vectorst.shape)
    # print(train_vectorst)
    # nsamples, nx, ny = train_vectorst.shape
    # train_vectors = train_vectorst.reshape((nsamples,nx*ny))
    # train_vectors=train_vectorst
    # print ("vector reshaped")
    # dense = train_vectors.todense()
    # denselist = dense.tolist()
    # feature_names = vectorizer.get_feature_names()
    # df = pd.DataFrame(denselist, columns=feature_names)
    # s = pd.Series(df.iloc[0])
    # print(s[s > 0].sort_values(ascending=False)[:10])

    test_vectors = vectorizer.transform(joinedTestCorpus)
    test_vectors = testCorpus

    joinedAppCorpus=np.array(joinedAppCorpus)
    joinedTestCorpus=np.array(joinedTestCorpus)
    appTarget=np.array(appTarget)
    testTarget=np.array(testTarget)
    # model = DBSCAN(eps=0.3,min_samples=10).fit(train_vectors)
    # print(train_vectors)
    model = KNeighborsClassifier(n_neighbors=2,algorithm="auto",weights='uniform').fit(train_vectors,appTarget)
    # model = MultinomialNB(alpha=0,fit_prior=False).fit(train_vectors, appTarget)

    return(model)

def eval(ev):
    joinedAppCorpus = []
    joinedTestCorpus = []
    tempTextCorpus = []
    model_list = []

    list_text = []
    list_target = []
    news_list = createNewsNew()
    display("=> List OK",'yellow')
    random.shuffle(news_list)
    news_list = news_list[:5000]
    for index,news in enumerate(news_list):
        news.clean_text()
        if (len(news.getCleanedText())>0):
            list_text.append(news.getCleanedText())
            list_target.append(news.getVeracity())
        if index%100==0 :
            print("News n°{} tagged".format(index))
    # padmax = 250
    # padmaxr = 300

    # for index,text in enumerate(list_text):
    #     if (len(text)<=padmax):
    #         tempTextCorpus.append(np.array(text+[[0]*padmaxr]*(padmax-len(text))))
    #     else:
    #         tempTextCorpus.append(np.array(text[:padmax]))
    #     if index%100==0 :
    #         print("News n°{} Padded".format(index))
    display("clean ok",'yellow')
    mid = 2*round((len(news_list)/3))
    appCorpus = list_text[:mid]
    testCorpus = list_text[mid:]
    appTarget = list_target[:mid]
    testTarget = list_target[mid:]
    for i in range(0,1):
        # vectorizers[i]=TfidfVectorizer()
        model_list.append(preprocessing(appCorpus,appTarget,testCorpus,testTarget))
        display("Model "+ str(i) +" : OK",'yellow')
    display("=>DONE Start of the evaluation ","yellow")

    # Global evaluation

    if ev:
        for array in appCorpus:
            joinedAppCorpus.append(' '.join(array))
        for array in testCorpus:
            joinedTestCorpus.append(' '.join(array))
        train_vectors = vectorizer.fit_transform(joinedAppCorpus)

        testTarget = np.array(testTarget)
        testTarget[testTarget=='mostly false']=int(0)
        testTarget[testTarget=='mostly true']=int(1)
        testTarget = [int(item) for item in testTarget]
        resList =[]
        for index,model in enumerate(model_list):
            test_vectors = vectorizer.transform(joinedTestCorpus)
            # test_vectorst = np.array(testCorpus)
            # print(test_vectorst.shape)
            # print(test_vectorst)
            # nsamples, nx, ny = test_vectorst.shape
            # test_vectors = test_vectorst.reshape((nsamples,nx*ny))
            # test_vectors=test_vectorst

            predicted = model.predict(test_vectors)
            predicted = np.array(predicted)
            predicted[predicted == 'mostly false']=0
            predicted[predicted == 'mostly true']=1
            predicted = [int(item) for item in predicted]
            resList.append(predicted)
            print("Model n°"+str(index)+" utilisé")
        resList = np.array(resList)
        pred = list(map(sum,zip(*list(resList))))

        # for index in range(len(resList[0])):
        #     temppred = 0
        #     temppred+=int(predict[index])
        #     pred[index]=sum[item[0] for item in reslist]
        #     print (temppred)
        #     if temppred<50:
        #         temppred=0
        #     else:
        #         temppred=1
        #     pred[index]=temppred


        display("Accuracy of the combined model = "+str(accuracy_score(testTarget,pred)),'yellow')

        confusion=confusion_matrix(testTarget, pred)

        matrice_confusion = pd.DataFrame(confusion, ["0","1"],
                      ["0","1"])
        precisionFalse = confusion[0][0]/(np.sum(confusion[0]))
        precisionTrue = confusion[1][1]/(np.sum(confusion[1]))
        display("Precision for False = "+str(precisionFalse),'yellow')
        display("Precision for True = "+str(precisionTrue),'yellow')
        pprint(matrice_confusion)
        print(train_vectors[0].shape)
        initial_type = [('float_input', TensorType([train_vectors[0].shape[0], train_vectors[0].shape[1]]  ))]
        onx = convert_sklearn(model_list[0],initial_types=initial_type)
        with open("Model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
    else:
        pickle.dump(model_list,open("finalized_model.sav",'wb'))
        pickle.dump(model_list,open("finalized_model.sav",'wb'))
    return model_list

def predict(link,model_list):
    start = time.time()
    article = getNewsFromXML(link)
    article.clean_text()
    text=article.getCleanedText()
    joined_Text = ' '.join(text)
    resList = []
    for index,model in enumerate(model_list):

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
    end = time.time()
    display("prediction ="+str(pred[0])+" % True","yellow")
    display("Prediction time = "+str(end-start),"yellow")
if len(sys.argv) == 2:
    model_list = eval(True)
else:
    model_list = eval(False)
