import NN_current
import numpy as np
import random
import argparse
import pickle
from ExtractData import Data
from math import log

# - - - - - - - - - - - - - - - -
# Main

# - - - - - - - - - - - - - - - -
# fonctions

def sigmoid(output):
	# Prevent overflow.
	output = np.clip(output, -500, 500 )
	output = 1.0/(1 + np.exp(-output))
	return output

def sigmoid_derivative(dE_da):
	return dE_da * (1. - dE_da)

def mean_squared(output, y):
	return 0.5*((output - y)**2)

def mean_squared_derivative(output, y):
	return y - output

def stable_softmax(output):
	z = output - max(output)
	num = np.exp(z)
	softmax = num/np.sum(num)
	return softmax

def softmax_derivative(sm):
	s = sm.reshape(-1,1)
	return np.diagflat(s) - np.dot(s, s.T)

def negative_log_likelihood(distrib, y):
	return -np.log(distrib[y])

def reLU(x):
	return x * (x > 0)
# - - - - - - - - - - - - - - - -
# Initialisation embeddings

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

#Extracting from the conlls
data = Data(word2vec, args.train, args.dev, args.test)


#engineering features
def engineer_features(numpy_from_conll):
	#before tag, before embedding, current embedding, after embedding
	numpy_engineered = np.zeros((len(numpy_from_conll), 168))
	#adding the one for NONE first tag
	numpy_engineered[0, 0] = 1
	numpy_engineered[1:len(numpy_from_conll), 18:68] = numpy_from_conll[:len(numpy_from_conll)-1, 0:50]
	numpy_engineered[:len(numpy_from_conll), 68:118] = numpy_from_conll
	numpy_engineered[:len(numpy_from_conll)-1, 118:168] = numpy_from_conll[1:len(numpy_from_conll), 0:50]
	return(numpy_engineered)


better_X_train_data = engineer_features(data.X_train)
better_X_dev_data = engineer_features(data.X_dev)
better_X_test_data = engineer_features(data.X_test)

# - - - - - - - - - - - - - - - -
# initialisation réseau

print("entrainer un réseau puis le tester sur le corpus test, tester un réseau déjà entrainé sur le corpus test ou test un réseau déjà entrainé sur une phrase ?")
print('\n')
print("Entrez 1 pour train+test, 2 pour test sur le corpus test, 3 pour test sur une phrase")
rep=input()
#cas où il y a une erreur de saisie
while(rep!='1' and rep!='2' and rep!='3'):
	print("tapez 1, 2 ou 3")
	rep=input()
#train + test
if(rep=='1') : 
	nb_neurons = [168, 50, 100, len(data.pos_semantics)]
	activation_function = sigmoid
	output_function = stable_softmax
	loss_function = mean_squared
	nn = NN_current.NeuralNetwork(nb_neurons, activation_function, output_function, loss_function)
	print("------------------------------")
	#on entraine le réseau
	nn.train(better_X_train_data, data.Y_train, epoch=2, mini_batch=1)
	print("nom du fichier pickle où enregistrer ce réseau entrainé ?")
	filename=input()
	#on enregistre le réseau au format pickle pour pouvoir le réutiliser sans avoir à ré entrainer
	pickle.dump(nn, open(filename, 'wb'))
	#on teste sur le corpus test
	print(nn.test(better_X_test_data, data.Y_test, data.testwords, data.pos_semantics, logs=False))
# test sur le corpus test avec un réseau déjà entrainé et enregistré au format pickle
if(rep=='2') :
	print("nom du fichier pickle où le réseau entrainé est enregistré ? ex : train_lemme")
	filename=input()
	#on charge le réseau entrainé
	nn=pickle.load(open(filename, 'rb'))
	#on teste sur le corpus test
	print(nn.test(better_X_test_data, data.Y_test, data.testwords, data.pos_semantics, logs=False))
# test sur une phrase rentrée par l'utilisateur avec un réseau déjà entrainé et enregistré au format pickle
if(rep=='3') : 
	print("nom du fichier pickle où le réseau entrainé est enregistré ? (train_words conseillé)")
	filename=input()
	#on charge le réseau entrainé
	nn=pickle.load(open(filename, 'rb'))
	#on entre la phrase que l'on veut tagger
	print("Tapez la phrase")
	phrase=input()
	X_test_phrase = np.zeros((len(phrase.split()), 50))
	Y_test_phrase = np.zeros((len(phrase.split()), len(data.pos_semantics)))
	for i in range (len(phrase.split())) :
		mot=""
		mot=phrase.split()[i]
		mot=mot.strip('.')
		if len(word2vec[mot]) == 0:
			X_test_phrase[i] = np.array(word2vec["chat"])
		else :
			X_test_phrase[i]=np.array(word2vec[mot])
	better_X_test_phrase = engineer_features(X_test_phrase)
	for i in range(len(phrase.split())) : 
		print(phrase.split()[i])
		print(data.pos_semantics[nn.predict(better_X_test_phrase[i])])
