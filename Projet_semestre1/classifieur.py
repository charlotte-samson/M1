#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import bz2
import os
import os.path

import sys
import argparse
import pickle
import random
import json

from objets import Element
from objets import Binome

#--------------------LECTURE DES ARGUMENTS--------------------

#usage_str=u"""__________________"""

usage_str=u"""Ce programme est un classifieur qui prend : 
un dictionnaire des binomes extraits au format pickle 
un dictionnaire content les traits sémantiques de binomes également au format pickle
et un dictionnaire contenant des prononciation de mot au format json"""

argparser = argparse.ArgumentParser(usage = usage_str)
argparser.add_argument('corpus_file', help='dictionnaire contenant des binômes au format pickle')
argparser.add_argument('semfeatures_file', help='dictionnaire contenant des traits sémantiques au format pickle')
argparser.add_argument('pronunciation_file', help='dictionnaire contenant des prononciation de mot au format json')
args = argparser.parse_args()



#----------CLASSE PERCEPTRON------------
class Perceptron():

	def __init__(self,corpus):
		self.params = defaultdict(lambda:defaultdict(lambda:int)) # à voir quel format on met aux paramètres
		self.keyfeatures=["anime","concret","general","positif","causalite","extralinguistique","logique","temporel","nombre","genre","syll","voy","ordrealpha"]
		for classe in range(1,11) :
			for feature in self.keyfeatures :
				self.params[classe][feature]=0
		self.lastParams = self.params # pour enregistrer la meilleure perf
	
		self.epoch = 0
		self.updates = 0
		self.perfTrain = 0
		self.oldPerfDev = -2
		self.perfDev = -1

		self.total=countOcc(corpus)

	def learn(self,train,dev):
	
		# tant que la perf en dev augmente et qu'on est pas à 100% sur le train
		while self.oldPerfDev < self.perfDev and self.perfTrain != 100:
				
			self.update(train) #mise à jour des paramètres

			#on check les perf en dev (pour voir si on doit s'arrêter)
			self.oldPerfDev = self.perfDev
			self.perfDev = self.evaluate(dev)[0]

			if(self.oldPerfDev>=self.perfDev): 
				self.params = self.lastParams #on récupère les anciens paramètres (ils étaient visiblement meilleurs)

			else :
				self.epoch+=1 #on le compte que là parce que si on s'arrête dans le if on veut pas faire +1
				print('\n',"Epoch ",self.epoch,'\n')
				print("classe exacte :  train acc = ",self.perfTrain[0]," dev acc = ",self.perfDev, " updates = ",self.updates)
				print("figé non figé : train =",self.perfTrain[1], " dev  = ",self.evaluate(dev)[1])
				print("figé alpha/figé inverse/non figé : train = ",self.perfTrain[2]," dev = ",self.evaluate(dev)[2])

	def classify(self, binome):
		# récupère les features du binôme (liste)
		feature = get_features(binome) 

		#on enregistre la prédiction qui a le produit scalaire le plus grand
		predBinome = "" # la classe
		valBinome = 0	# son produit scalaire

		res=[]
		i=0
		r=0

		for classe in self.params.keys() :
			for feat in feature : 
				if feat in self.params[classe] : 
					r=r+(feature[feat]*self.params[classe][feat])
				else :
					self.params[classe][feat]=0
			res.append((classe,r))
		predBinome,valBinome= sorted(res,key=lambda x: x[1], reverse=True)[0]
		return predBinome

	def update(self, listeBinomes):
		self.updates = 0

		total = 0
		bons = 0
		figénonfigé = 0 #nb de binome pour lesquels on a bien prédit si c'était figé ou pas 
		#2 classes : figé = [1,2,9,10] non figé = [3,4,5,6,7,8]

		figéordre=0 #nb de binome pour lesquels on a bien prédit si c'était figé ou pas et dans quel sens
		#3 classes : figé dans le sens alpha = classe [9,10], figé dans le sens inverse = classe [1,2], non figé [3,4,5,6,7,8]

		random.shuffle(listeBinomes) # ne pas apprendre toujours dans le même ordre

		for binome in listeBinomes: # pour chaque binôme
			prediction = self.classify(binome) # on effectue le classifie

			if prediction != binome.classe : # si c'était mal classifié
				self.updates += 1

				observation = get_features(binome) # on récupère le vecteur qui servira à faire la mise à jour sur les paramètres

				# MISE A JOUR DES PARAMETRES :
				for feat in observation.keys():	#vecteur de la mauvaise prediction -= observation
					self.params[prediction][feat] -= observation[feat]*(binome.totalOcc/self.total)
					#pour pondérer par la fréquence d'apparition du binome

				for feat in observation.keys(): 	#vecteur de la bonne preduction += observation
					self.params[binome.classe][feat] += observation[feat]*(binome.totalOcc/self.total)
					#pour pondérer par la fréquence d'apparition du binome

			else :
				bons+=1
				if ((binome.classe in [1,2,9,10] and prediction in [1,2,9,10] ) or (binome.classe in [3,4,5,6,7,8] and prediction in [3,4,5,6,7,8])) :
					figénonfigé+=1
				if ((binome.classe in [9,10] and prediction in [9,10] )or (binome.classe in [1,2] and prediction in [1,2]) or (binome.classe in [3,4,5,6,7,8] and prediction in [3,4,5,6,7,8])) :
					figéordre+=1

		self.perfTrain = ((bons/len(listeBinomes))*100,(figénonfigé/len(listeBinomes)*100),(figéordre/len(listeBinomes)*100)) # mise à jour de la perf en train (au cas où elle soit à 100%, dans ce cas on arrête)

	def evaluate(self, listeBinomes):
		bon = 0	#nb de binomes où la bonne classe exacte a été donné 
		figénonfigé = 0 #nb de binome pour lesquels on a bien prédit si c'était figé ou pas 
		#2 classes : figé = [1,2,9,10] non figé = [3,4,5,6,7,8]

		figéordre=0 #nb de binome pour lesquels on a bien prédit si c'était figé ou pas et dans quel sens
		#3 classes : figé dans le sens alpha = classe [9,10], figé dans le sens inverse = classe [1,2], non figé [3,4,5,6,7,8]

		for binome in listeBinomes: # pour chaque binôme
			prediction = self.classify(binome) # on le classifie
			bon+= 1 if prediction == binome.classe else 0 # on compte un bon si c'est réussi
			if ((binome.classe in [1,2,9,10] and prediction in [1,2,9,10] ) or (binome.classe in [3,4,5,6,7,8] and prediction in [3,4,5,6,7,8])) :
				figénonfigé+=1
			if ((binome.classe in [9,10] and prediction in [9,10] )or (binome.classe in [1,2] and prediction in [1,2]) or (binome.classe in [3,4,5,6,7,8] and prediction in [3,4,5,6,7,8])) :
				figéordre+=1
			
				
		return ((bon/len(listeBinomes)*100),(figénonfigé/len(listeBinomes)*100),(figéordre/len(listeBinomes)*100))

#----------EXTRACTION DES FEATURES----------

def nb_syll(mot) :
	#1+nb de . dans transcription
	nb=1
	transcription=pronunciation[mot][0]
	for c in transcription :
		if(c=='.') :
			nb=nb+1
	return nb

def voyelle_longue(mot):
	#renvoie True s'il y a une voyelle allongée dans un mot présent dans le fichier json
	return ':' in pronunciation[mot][0]

def countOcc(dico) :
	#compte le nombre de binomes dans le corpus
	for key in corpus : 
		total=sum(corpus[key])
	return total

def get_features(binome) :
	#grâce au fichier pkl : animé/inanimé, positif/négatif, concret/abstrait, +général/-général, causalité, extralinguistique, ordrelogique, ordre temporel
	#grâce au fichier conll : singulier/pluriel, masculin/féminin 
	#grâce au fichier json : -de syllabes/+ de syllabes,voyelle courte/voyelle longue
	#12 features ( 7 sémantiques, 2 grammaticaux, 2 phonétique, 1 orthographique)
	
	#keyfeatures=["anime","concret","general","positif","causalite","extralinguistique","logique","temporel","nombre","genre","syll","voy"]

	features=defaultdict(int)

	felt1=binome.elt1.forme
	felt2= binome.elt2.forme
	
	#si le binome a été annoté et qu'il est donc dans semfeatures
	if((felt1,felt2) in semfeatures) :
		#si on a bien l'ordre animé/inanimé
		if(semfeatures[(felt1,felt2)]['features'][felt1]['animate']['meta']=='yes' and semfeatures[(felt1,felt2)]['features'][felt2]['animate']['meta']=='no' ) :
			#feature animé = 1
			features["anime"] = 1
		else :
			features["anime"] = 0
		#si on a bien l'ordre concret abstrait
		if(semfeatures[(felt1,felt2)]['features'][felt1]['abstract']['meta']=='no' and semfeatures[(felt1,felt2)]['features'][felt2]['animate']['meta']=='yes' ) :
			#feature concret = 1	
			features["concret"] = 1
		else :
			features["concret"] = 0
		if(felt1+'  est plus générale que  '+felt2 in semfeatures[(felt1,felt2)]['features']) :
		#si on a bien l'ordre +général/-général
			if(semfeatures[(felt1,felt2)]['features'][felt1+'  est plus générale que  '+felt2]['meta']=='no') :
				features["general"] = 1
			else :
				features["general"] = 0
		#si on a bien l'ordre positif/négatif
		if(semfeatures[(felt1,felt2)]['features'][felt1]['positive']['meta']=='yes' and semfeatures[(felt1,felt2)]['features'][felt2]['positive']['meta']=='no' ) :
			features["positif"] = 1
		else :
			features["positif"] = 0
		#si on a bien l'ordre de causalité
		if(semfeatures[(felt1,felt2)]['features']['relation']['causality']['meta']=='yes') :
			features["causalite"] = 1
		else :
			features["causalite"] = 0
		#si on a bien l'ordre extralinguistique
		if(semfeatures[(felt1,felt2)]['features']['relation']['extra_linguistic']['meta']=='yes'):
			features["extralinguistique"] = 1
		else :
			features["extralinguistique"] = 0
		#si on a bien l'ordre logique
		if(semfeatures[(felt1,felt2)]['features']['relation']['logical_order']['meta']=='yes'):
			features["logique"] = 1
		else :
			features["logique"] = 0
		#si on a bien l'ordre temporel
		if(semfeatures[(felt1,felt2)]['features']['relation']['temporal_order']['meta']=='no'):
			features["temporel"] = 1
		else :
			features["temporel"] = 0

	
	#si les 2 mots du binomes ont bien l'attribut nombre 
	if (hasattr(binome.elt1,'nb') and hasattr(binome.elt2,'nb')) :
	#si on a bien l'ordre singulier/pluriel
		if(binome.elt1.nb=='s' and binome.elt2.nb == 'p') :
			features["nombre"] = 1
		else :
			features["nombre"] = 0
	
	#si les 2 mots du binomes ont bien l'attribut genre 
	if (hasattr(binome.elt1,'genre') and hasattr(binome.elt2,'genre')) :
		#si on a bien l'ordre masculin/féminin
		if(binome.elt1.genre=='m' and binome.elt2.genre=='f') :
			features["genre"] = 1
		else :
			features["genre"] = 0

	#si on a bien la transcription du mot dans le dico extrait du fichier json
	if(binome.elt1.forme in pronunciation and binome.elt2.forme in pronunciation):
		if(pronunciation[binome.elt1.forme] and pronunciation[binome.elt2.forme]) :
			#+ de syllabe - de syllabe
			if(nb_syll(binome.elt1.forme) < nb_syll(binome.elt2.forme)) :
				features["syll"] =1
			else :
				features["syll"]=0
			
			#voyelle courte/voyelle longue
			if((voyelle_longue(binome.elt1.forme) == False) and (voyelle_longue(binome.elt2.forme)))  :
				features["voy"]=1
			else :
				features["voy"]=0
	
	#proportion du binome dans l'ordre alphabétique
	features["ordrealpha"]=(corpus[binome][0]+corpus[binome][1]+corpus[binome][4]+corpus[binome][5])/binome.totalOcc
	
	return dict(features)

#########################################
#------------MAIN------------------------
#########################################

# CONSTITUTION DES CORPUS TRAIN - DEV - TEST
corpus = pickle.load(open(args.corpus_file,'rb')) # dico : k = Binome v = [a,b,c,d,e,f,g,h]

keys = list(corpus.keys()) # on récupère que les clés (leur classe est contenue dans l'objet binôme)
random.shuffle(keys)

train = keys [:(int)(len(keys)*0.9)]
dev = keys [(int)(len(keys)*0.9)+1:(int)(len(keys)*0.95)]
test = keys [(int)(len(keys)*0.95)+1:]

semfeatures = pickle.load(open(args.semfeatures_file,'rb'))

with open(args.pronunciation_file, 'r') as f:
    pronunciation = json.load(f)

# --------------Perceptron---------------------
classifieur = Perceptron(corpus)
classifieur.learn(train, dev)
print("\n","classe exacte : ",classifieur.evaluate(test)[0])
print("figé/non figé : ",classifieur.evaluate(test)[1]) 
print("figé alpha/figé inverse/non figé : ",classifieur.evaluate(test)[2])