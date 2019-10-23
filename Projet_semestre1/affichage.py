#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

from collections import defaultdict
from math import *
import pprint as pp

import sys
import argparse
import pickle

from objets import Binome
from objets import Element

#-----------------LECTURE DES ARGUMENTS------------------------#

usage_str=u""" Ce programme affiche les représentations graphiques des comptes effectués
				sur les binômes extraits des corpus précédemment. """

argparser = argparse.ArgumentParser(usage = usage_str)
argparser.add_argument('pkl_file', help = 'Fichier pkl contenant le dico de binômes')
argparser.add_argument('save_file', help='Fichier où enregistrer le dico nettoyé')
args = argparser.parse_args()


#----------------FONCTIONS POUR LA DESCRIPTION-----------------#

def afficheComptes(dico):
	#On est sur dictionnaire de la forme créée par extraction:
	# 	clé = Binome (mot1 mot2 cat classe) 
	# 	val = [a, b, c, d, e, f, g, h]

	direct,alpha,et,fige,fige_alpha = 0,0,0,0,0

	for binome in dico:
		comptes = dico[binome] #on récupère dans une liste les comptes
		direct += sum(x for x in comptes[:4])
		alpha += (comptes[0]+comptes[1]+comptes[4]+comptes[5])
		et += (comptes[0]+comptes[2]+comptes[4]+comptes[6])

		if binome.classe in [1,2,9,10] :
			fige += 1
			if binome.classe in [1,2]:
				fige_alpha +=1

	print("binomes = ", len(dico))
	print("total =", total(dico))
	print("direct =",direct)
	print("alpha = ", alpha)
	print("et = ", et)
	print("figés = ", fige, "/", len(dico), "dont ", fige_alpha, "dans l'ordre alpha.")

def afficheGraphiques(dico):
	'''
		ARGUMENTS :
			- dico : dictionnaire clé = binome val = liste [a,b,c,d,e,f,g,h]
			- apppMin : le nombre d'apparitions minimum du binôme pour le prendre en compte
	'''

	datas = {} #dictionnaire de dictionnaires 

	datas["comptes_etou"] = defaultdict(int) #population = occurences
	datas["comptes_fige"] = defaultdict(int)
	datas["comptes_alpha"] = defaultdict(int)
	datas["comptes_classes"] = defaultdict(int) #population = binomes

	for binome in dico :

		comptes = dico[binome]

		datas["comptes_etou"]["et"] += comptes[0]+comptes[2]+comptes[4]+comptes[6]
		datas["comptes_etou"]["ou"] += comptes[1]+comptes[3]+comptes[5]+comptes[7]

		datas["comptes_alpha"]["alpha"] += comptes[0]+comptes[1]+comptes[4]+comptes[5]
		datas["comptes_alpha"]["inverse"] += comptes[2]+comptes[3]+comptes[6]+comptes[7]
			
		datas["comptes_classes"][binome.classe] += 1

		if(binome.classe in [1,2,9,10]) :
			datas["comptes_fige"]["fige"] +=1
		else:
			datas["comptes_fige"]["non_fige"] +=1
		
	for key, val in datas.items():
		pp.pprint(val)
		showHistogramme(val,key)

def showHistogramme(dico_comptes, title):

	dico_distrib = makeDistrib(dico_comptes)

	data= []
	
	data.extend({"x": x, "y" : y} for x, y in dico_distrib.items())
	
	data = pd.DataFrame(data)

	ax = sns.barplot(x="x", y="y", data=data)

	plt.suptitle(title)

	plt.ylim(0,100)
	plt.ylabel("%")
	plt.xlabel("")

	plt.show()

def makeDistrib(dico):
	total = sum (x for x in dico.values())
	for key in dico.keys():
		dico[key] = (dico[key]/total)*100
	return dico

def total(dico):
	total = 0
	for binome in dico:
		total+= binome.totalOcc
	return total

# -------------FONCTIONS POUR LE NETTOYAGE---------------------#

def findAppMax(dico_binomes):
	# 1 - voir combien il y a de binômes avant combien d'occurrences
	dico_occ = defaultdict(int) # cle = totalOcc, val = nb de binomes ayant ce totalOcc
	for binome in dico_binomes :
		dico_occ[binome.totalOcc]+=1

	#pp.pprint(dico_occ)
	#showHistogramme(dico_occ, "apparitions des binômes") #attention plante avec le wiki, trop de données
	
	liste_occ = sorted(list(dico_occ.keys())) #on trie pour prendre dans l'ordre

	seuil_max = len(dico_binomes)*0.999 #le numéro du binôme jusqu'auquel on va prendre
	stop = len(dico_binomes) # compteur (qu'on va baisser petit à petit)
	indice = len(liste_occ)-1 #l'indice du nombre d'occurences dans liste_occ
	
	while(stop>seuil_max): #tant qu'on a pas enlevé le bon nombre de binômes on continue à descendre dans la liste
		stop -= dico_occ[liste_occ[indice]]
		indice -= 1

	return liste_occ[indice] #on retourne le nombre max d'occurrences

def nettoieDonnees(dico_sale,appMin,appMax):

	dico_nettoye = {}

	for binome in dico_sale:
		if(binome.totalOcc>=appMin and binome.totalOcc<= appMax):
			dico_nettoye[binome] = dico_sale[binome]

	return dico_nettoye


############################MAIN##################################

# Récupération des données de l'extraction.
dico_binomes = pickle.load(open(args.pkl_file,'rb'))

#########"""AVANT NETTOYAGE"""########

print("Voici le contenu de l'extraction avant nettoyage : \n")
#Affiche les comptes sans nettoyage
afficheComptes(dico_binomes)

print("\nOn peut y effectuer les comptes suivants : \n")
# Affiche les graphiques sans nettoyer les données au préalable
afficheGraphiques(dico_binomes)


########## """NETTOYAGE""" ##########
#On trouve la borne supérieure pour le second nettoyage
appMax = findAppMax(dico_binomes)
print("\nLe nombre d'occurences maximales représentatices pour un binôme est fixé à ",appMax,"pour ce corpus")

dico_binomes_nettoye = nettoieDonnees(dico_binomes,3,appMax)

######### """APRES NETTOYAGE""" #########

print("\nVoici le contenu de l'extraction après nettoyage : \n")
# Affiche les comptes  une fois le nettoyage fait
afficheComptes(dico_binomes_nettoye)

print("\nOn peut y effectuer les comptes suivants : \n")
# Affiche des graphiques en ayant nettoyé les données au préalable
afficheGraphiques(dico_binomes_nettoye)

findAppMax(dico_binomes_nettoye)

pickle.dump(dict(dico_binomes_nettoye), open(args.save_file+".pkl", "wb"))
