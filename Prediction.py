from News import *
from Color import *
from XML2News import *
from sklearn.feature_extraction.text import TfidfTransformer
import sys
import numpy as np
import pickle
def predict(link):
    model_list = pickle.load(open("finalized_model.sav","rb"))
    vectorizers = pickle.load(open("finalized_vectorizers.sav",'rb'))
    article = getNewsFromXML(link)
    article.clean_text()
    text=article.getCleanedText()
    joined_Text = ' '.join(text)
    resList = []
    for index,model in enumerate(model_list):
        test_vectors = vectorizers[index].transform([joined_Text])
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
    display("prediction = "+str(pred[0]*10)+" % True","yellow")
if len(sys.argv)!=2:
    error("Wrong Number of argument\nUsage : Prediction.py path/to/file.xml")
else:
    predict(sys.argv[1])
