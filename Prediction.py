from News import *
from Color import *
from XML2News import *
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import onnxruntime as rt
import re
from newspaper import Article

from flask import make_response, abort, jsonify
import base64
sess = rt.InferenceSession("Model.onnx")
def valider(url):
    """
    This function creates a new person in the people structure
    based on the passed in person data

    :param person:  person to create in people structure
    """

    link = base64.b64decode(url).decode("utf-8")
    score = predict(link,"url")
    print("SCORE " + str(score))
    return jsonify(score)

def predict(link,mode):

    if mode == "corpus":
        article = link
    if mode == "url":
        toi_article = Article(link, language="en")
        toi_article.download()
        toi_article.parse()
        toi_article.nlp()
        text = toi_article.text
        article = News("author",text,"links","orientation","unk","title")
    if mode == "eval":
        article = News("author",link,"links","orientation","unk","title")
    article.clean_text()
    text=article.getCleanedText()
    joined_Text = np.array([' '.join(text)])

    resList =[]
    #prediction for each model of our list

    resList = np.array(resList)
    output_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name
    pred= sess.run([output_name], {input_name: joined_Text})
    print(pred)

    display("prediction = "+str(pred[0])+" % True","yellow")
    if pred[0][0]=='mostly false':
        return(0)
    return (1)
