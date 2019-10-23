import NN_current
import numpy as np
import random
import argparse
import pickle
from ExtractData import Data
from math import log
#import scikitplot as skplt
import matplotlib.pyplot as plt
from collections import defaultdict
import pylab

usage_str=u""" Ce programme necessite 1 fichier pickle de dictionnaire word2vec ainsi 3 fichiers conll annotes (train dev test) contenant des mots annotés
"""

argparser = argparse.ArgumentParser(usage = usage_str)
argparser.add_argument('word2vec', help='fichier pickle de dictionnaires des embeddings en format word2vec')
argparser.add_argument('train', help='fichier annoté au format CoNLL, corpus train')
argparser.add_argument('dev', help='fichier annoté au format CoNLL,corpus dev')
argparser.add_argument('test', help='fichier annoté au format CoNLL,corpus test')
args = argparser.parse_args()

#Loading the word to vec
with open(args.word2vec, 'rb') as f:
	word2vec = pickle.load(f)
print(word2vec["remarquable"])
print(len(word2vec["un"]))

#Extracting from the conlls
data = Data(word2vec, args.train, args.dev, args.test)

graphe=plt.figure(figsize=(20,15))
pos2countwordstrain=defaultdict(int)
pos2countwordsdev=defaultdict(int)
pos2countwordstest=defaultdict(int)
sommetrain=sum([tag for tag in np.asarray(data.Y_train)])
sommedev=sum([tag for tag in np.asarray(data.Y_dev)])
sommetest=sum([tag for tag in np.asarray(data.Y_test)])
for i in range(18) :
	pos2countwordstrain[data.pos_semantics[i]]=sommetrain[i]
	pos2countwordsdev[data.pos_semantics[i]]=sommedev[i]
	pos2countwordstest[data.pos_semantics[i]]=sommetest[i]

for k,v in pos2countwordstrain.items() :
	pos2countwordstrain[k]=(v/len(data.Y_train))*100
for k,v in pos2countwordsdev.items() :
	pos2countwordsdev[k]=(v/len(data.Y_dev))*100
for k,v in pos2countwordstest.items() :
	pos2countwordstest[k]=(v/len(data.Y_test))*100

corpus = ['train', 'dev', 'test']
print(pos2countwordstrain)
print(pos2countwordsdev)
print(pos2countwordstest)
NOUN=[]
NOUN.append(pos2countwordstrain["NOUN"])
NOUN.append(pos2countwordsdev["NOUN"])
NOUN.append(pos2countwordstest["NOUN"])
ADP=[]
ADP.append(pos2countwordstrain["ADP"])
ADP.append(pos2countwordsdev["ADP"])
ADP.append(pos2countwordstest["ADP"])
VERB=[]
VERB.append(pos2countwordstrain["VERB"])
VERB.append(pos2countwordsdev["VERB"])
VERB.append(pos2countwordstest["VERB"])
DET=[]
DET.append(pos2countwordstrain["DET"])
DET.append(pos2countwordsdev["DET"])
DET.append(pos2countwordstest["DET"])
PUNCT=[]
PUNCT.append(pos2countwordstrain["PUNCT"])
PUNCT.append(pos2countwordsdev["PUNCT"])
PUNCT.append(pos2countwordstest["PUNCT"])
SYM=[]
SYM.append(pos2countwordstrain["SYM"])
SYM.append(pos2countwordsdev["SYM"])
SYM.append(pos2countwordstest["SYM"])
ADJ=[]
ADJ.append(pos2countwordstrain["ADJ"])
ADJ.append(pos2countwordsdev["ADJ"])
ADJ.append(pos2countwordstest["ADJ"])
PROPN=[]
PROPN.append(pos2countwordstrain["PROPN"])
PROPN.append(pos2countwordsdev["PROPN"])
PROPN.append(pos2countwordstest["PROPN"])
PRON=[]
PRON.append(pos2countwordstrain["PRON"])
PRON.append(pos2countwordsdev["PRON"])
PRON.append(pos2countwordstest["PRON"])
X=[]
X.append(pos2countwordstrain["X"])
X.append(pos2countwordsdev["X"])
X.append(pos2countwordstest["X"])
INTJ=[]
INTJ.append(pos2countwordstrain["INTJ"])
INTJ.append(pos2countwordsdev["INTJ"])
INTJ.append(pos2countwordstest["INTJ"])
NUM=[]
NUM.append(pos2countwordstrain["NUM"])
NUM.append(pos2countwordsdev["NUM"])
NUM.append(pos2countwordstest["NUM"])
CONJ=[]
CONJ.append(pos2countwordstrain["CONJ"])
CONJ.append(pos2countwordsdev["CONJ"])
CONJ.append(pos2countwordstest["CONJ"])
AUX=[]
AUX.append(pos2countwordstrain["AUX"])
AUX.append(pos2countwordsdev["AUX"])
AUX.append(pos2countwordstest["AUX"])
SCONJ=[]
SCONJ.append(pos2countwordstrain["SCONJ"])
SCONJ.append(pos2countwordsdev["SCONJ"])
SCONJ.append(pos2countwordstest["SCONJ"])
PART=[]
PART.append(pos2countwordstrain["PART"])
PART.append(pos2countwordsdev["PART"])
PART.append(pos2countwordstest["PART"])
ADV=[]
ADV.append(pos2countwordstrain["ADV"])
ADV.append(pos2countwordsdev["ADV"])
ADV.append(pos2countwordstest["ADV"])
ind = [x for x, _ in enumerate(corpus)]
add=[np.array(NOUN),np.array(ADP),np.array(DET),np.array(PUNCT),np.array(VERB),np.array(PROPN),np.array(ADJ),np.array(PRON),np.array(ADV),np.array(NUM),np.array(CONJ),np.array(AUX),np.array(SCONJ),np.array(PART),np.array(X),np.array(SYM),np.array(INTJ)]
p1=plt.bar(ind, NOUN , width=0.8, label='NOUN', color='blue',bottom=sum(add[1:]))
p2=plt.bar(ind, ADP, width=0.8, label='ADP', color='black',bottom=sum(add[2:]))
p3=plt.bar(ind, DET, width=0.8, label='DET', color='red',bottom=sum(add[3:]))
p4=plt.bar(ind,PUNCT , width=0.8, label='PUNCT', color='yellow',bottom=sum(add[4:]))
p5=plt.bar(ind, VERB, width=0.8, label='VERB', color='green',bottom=sum(add[5:]))
p6=plt.bar(ind, PROPN, width=0.8, label='PROPN', color='grey',bottom=sum(add[6:]))
p7=plt.bar(ind, ADJ, width=0.8, label='ADJ',bottom=sum(add[7:]))
p8=plt.bar(ind, PRON, width=0.8, label='PRON',bottom=sum(add[8:]))
p9=plt.bar(ind, ADV, width=0.8, label='ADV',bottom=sum(add[9:]))
p10=plt.bar(ind, NUM, width=0.8, label='NUM',bottom=sum(add[10:]))
p11=plt.bar(ind, CONJ, width=0.8, label='CONJ',bottom=sum(add[11:]))
p13=plt.bar(ind, AUX, width=0.8, label='AUX',bottom=sum(add[12:]))
p14=plt.bar(ind,SCONJ , width=0.8, label='SCONJ',bottom=sum(add[13:]))
p15=plt.bar(ind, PART, width=0.8, label='PART',bottom=sum(add[14:]))
p16=plt.bar(ind, X, width=0.8, label='X',bottom=sum(add[15:]))
p17=plt.bar(ind,SYM , width=0.8, label='SYM',bottom=sum(add[16:]))
p18=plt.bar(ind, INTJ, width=0.8, label='INTJ')
plt.xticks(ind, corpus,fontsize=18)
plt.yticks(fontsize=18)
plt.legend(bbox_to_anchor=(1., 1.),ncol=1,fontsize=20)
plt.ylabel("Pourcentage d'apparition dans le corpus",fontsize=22)
plt.xlabel("corpus",fontsize=22)
plt.title("Répartition des PoS selon les corpus",fontsize=28)
plt.ylim=1.0
# rotate axis labels
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
plt.savefig("proportiontagparcorpus2.png")

"""
y_true = data.Y_dev
# ground truth labels
y_probas = 
# predicted probabilities generated by sklearn classifier
#y_pred.reshape(len(y_pred),1)
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()
"""