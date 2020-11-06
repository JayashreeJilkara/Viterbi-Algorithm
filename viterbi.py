import sys
import pandas as pd
import nltk
import re
from nltk import bigrams

def dataSet(Path):
    with open(Path,'r') as text:
        List_Of_Words=[]
        List_Of_Tags=[]
        Overall_List = []
        for x in text.readlines():
            for y in x.split():
                term, sym, pos_Tag = y.partition('/')
                Overall_List.append((term,pos_Tag))
                List_Of_Words.append(term)
                List_Of_Tags.append(pos_Tag)
    return Overall_List,List_Of_Words,List_Of_Tags

train_set,train_word_List, train_tag_List = dataSet('POS.train')
test_set, test_word_List, test_tag_List = dataSet('POS.test')


train_setTokenCount ={}
for z in train_set:
    if(z not in train_setTokenCount):
        train_setTokenCount[z]=1
    else:
        train_setTokenCount[z]+=1

#print(train_setTokenCount)

train_setTagCount = {}
for z in train_tag_List:
    if(z not in train_setTagCount):
        train_setTagCount[z]=1
    else:
        train_setTagCount[z]+=1

#print(train_setTagCount)

bi_tokens = bigrams(train_tag_List)
trainBiTagCount = {}
for z in bi_tokens:
    if(z not in trainBiTagCount):
        trainBiTagCount[z]=1
    else:
        trainBiTagCount[z]+=1

#print(trainBiTagCount)

with open('POS.test','r') as data:
    text1 = []
    for i in data.readlines():
        text = []
        for j in i.split():
            term, sym, pos_Tag = j.partition('/')
            text.append((term))
        text1.append(text)
    testWordSet = (text1)

#print(testWordSet)

def ViterbiAlgorithm(token, List_Of_Tags):
    s = []

    for x, y in enumerate(token):
        z = []
        for t in List_Of_Tags:
            if x == 0:
                try:
                    trans_Prob = train_setTokenCount[('.', t)] / train_setTagCount[t]
                except:
                    trans_Prob = 1 / len(trainBiTagCount)
            else:
                try:
                    trans_Prob = train_setTokenCount[(s[-1], t)] / train_setTagCount[t]
                except:
                    trans_Prob = 1 / len(trainBiTagCount)
            try:
                # print(train_setTokenCount[(word,tag)])
                # print(train_setTagCount[tag])
                emissionProb = train_setTokenCount[(y, t)] / train_setTagCount[t]
            except:
                emissionProb = 1 / len(train_setTokenCount)
            score = emissionProb * trans_Prob
            z.append(score)
        m = max(z)
        smax = List_Of_Tags[z.index(m)]
        s.append(smax)
    return list(zip(token, s))


#print(testWordSet)
#print(train_setTokenCount)
TagList = list(train_setTagCount.keys())
predList = []

for i in testWordSet:
    predList+= ViterbiAlgorithm(i, TagList)

df= pd.DataFrame(predList,columns=['Word','tag'])
#print(df)
df.to_csv('POS.test.out')

#print(systemPrediction)
#print(test_set)

#word_check = [m for m, n in zip(systemPrediction, test_set) if m == n]
#print(len(word_check))
#accuracy = len(word_check)/len(systemPrediction)
#print(accuracy*100)


#wrongWords = [n for m, n in enumerate(zip(predList, test_set)) if n[0] != n[1]]
#print(len(wrongWords))
#print(wrongWords)

print(predList)

r = 0
w = 0
for i in range(len(test_set)):
  a = test_set[i]
  b = predList[i]
  for z in range(len(a)):
    if a[z] == b[z]:
      r = r+1
    else:
      w = w +1

print('Accuracy on trainset is: ',(r/(r+w))*100,'%')
#print('Loss is: ',(w/(r+w))*100,'%')