from Prediction import *
from XML2News import *

news_list= createNews()
print("Liste crée")
res=0
for news in news_list:
    pred,len=predict(news,"corpus")
    print("Score predicted : "+pred+"/"+str(len)+" | real class: "+ news.getVeracity())
    if int(pred)<=50 and news.getVeracity()=="mostly false":
        res+=1
    elif int(pred)>50 and news.getVeracity()=="mostly true":
        res+=1

print("Score total = "+str(res))
