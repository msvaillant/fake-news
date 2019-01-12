#!/usr/bin/python3
import json
import sys
class Text2Vect(object):
    """docstring for Text2Vect."""
    def __init__(self, words,text):
        self.listWords = []

        f = open(words, 'r',errors="replace")
        fArticle = open(text,'r',errors="replace")
        self.rawText = fArticle.read()
        rawWords = f.read()
        jsondata = json.loads(rawWords)
        for category in jsondata:
            self.listWords.extend(jsondata[category])
        self.resVector = [0] * len(self.listWords)
    def countOccurence(self):
        for index,word in enumerate(self.listWords):
            self.resVector[index] = self.rawText.count(word)
        resFile = open('ResultVector','a+')
        resFile.write(str(self.resVector)+'\n')

if (len(sys.argv) != 3):
    print("Wrong number of argument ("+str(len(sys.argv)-1)+")\n")
    print("=> ./Text2Vect Wordfile Text")
    exit(0)
else:
    text2vec = Text2Vect(sys.argv[1],sys.argv[2])
    print("Processing Article " + str(sys.argv[2]) + "\n")
    text2vec.countOccurence()
