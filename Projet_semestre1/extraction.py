#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import bz2
import os
import os.path
import glob 
import string
import sys
import argparse
import pickle

from objets import Element
from objets import Binome

#--------------------LECTURE DES ARGUMENTS--------------------

usage_str=u""" Ce programme extrait les binômes dans un corpus annoté au format CoNLL
           (une phrase par paragraphe, un mot par ligne avec des informations par mots
           la ligne, séparés par des espaces.)Ces annotations sont du type morphologique et syntaxique.
           Il les enregistre ensuite dans un fichier texte.
"""

argparser = argparse.ArgumentParser(usage = usage_str)
argparser.add_argument('rootd', help='Dossier de fichiers annotés au format CoNLL')
argparser.add_argument('save_file', help='Fichier ou sont enragistrés les binômes extraits')
argparser.add_argument('methode', help='lineaire ou dependance : la méthode d\'extraction')
args = argparser.parse_args()

#----------------METHODES----------------

def isLatin(mot) :
	# mot est de type string
	# la fonction renvoie un booleen
	# 		true si le mot est composé uniquement de caractères latins
	# 		false sinon
	try :
		mot.encode(encoding='utf-8').decode('ascii')
	except UnicodeDecodeError :
		return False
	else : 
		return True# fonction auxilliaire pour le nettoyage

def remove_endpunctuation(mot):
	# mot est de type string
	# cette fonction renvoie le mot en supprimant la ponctuation finale
    return mot[:-1]

def has_punctuation(mot) :
	# mot est de type string
	# cette fonction renvoie un boolean
	#		true si ce mot a un caractère de ponctuation à la fin
	#		false sinon
    for carac in mot : 
        if (carac in string.punctuation) :
            return True
    return False

def has_endpunctuation(mot) :
	# mot est de type string
	# cette fonction renvoie un boolean
	#		true si ce mot a un caractère de ponctuation à la fin
	#		false sinon
    if ((has_punctuation(mot[:-1])==False) and (mot[len(mot)-1] in string.punctuation)):
    	return True
    return False

def isBinome(elt1, elt2):
	# fonction de nettoyage
	# arguments : 2 éléments
	# renvoie un booleen
	# 		true si ces deux éléments forment un binôme pertinent
	#		false sinon
	if (elt2[3] != elt1[3]) : #s'ils sont pas de la même catégorie
		return False
	if (elt1[3] == "PONCT") : # si la catégorie est ponct
		return False
	if (elt1[1].isdigit() or elt2[1].isdigit()) : # si un des deux mots est un nombre (! attention prends pas en compte les nb à virgules)
		return False
	if (isLatin(elt1[2]) == False or isLatin(elt2[1]) == False) : # si les caractères ne sont pas latins
		return False
	if(has_punctuation(elt1[1]) or has_punctuation(elt2[1])) :
		return False
	return True#NETTOYAGE

def extractDependance(filename):
	''' Extrait par dépendances les binômes présents dans le fichier source et les enregistre dans le dictionnaire "binomes"

	*INPUT :
		filename (String) _ un  fichier conll avec des phrases
	*OUTPUT :
		void, remplit binomes dans le dictionnaire binomes
	'''
	with open (filename,"r") as stream :
		#print(filename)

		sentence = [] #liste qui contient à tour de rôle les phrases

		for line in stream :
			infos=line.strip().split()
			if(infos != []):

				sentence.append(infos) #on les ajoute à la phrase tant qu'on a pas de ligne vide
			
			else : #on est sur une ligne vide => fin de la phrase
				for infos in sentence : #pour chacun des mots
					
					if (infos[1] == "et" or infos[1] == "ou") : #on regarde si c'est "et" ou "ou"	
						
						try :

							for lidxgouv in [int(x)-1 for x in infos[6].split("|")] : # pour chaque élément gouverné par la coordination
									
								elt1 = sentence[lidxgouv] #récupère son gouverneur(s)
								
								for i in range((int(infos[0])-1)+1, len(sentence)) : #cherche son/ses gouvernés
									
									try :  # parfois on a des problèmes de valeurs donc on est obligés de mettre un try
										
										if(int(infos[0])-1 in [int(x)-1 for x in sentence[i][6].split("|")]): #si la coordination le gouverne
											
											elt2 = sentence[i] 
										
											if (isBinome(elt1,elt2)): # NETTOYAGE
												b1 = Element(elt1)
												b2 = Element(elt2)

												# BINOME DIRECT
												if(b1.lidx+1 == int(infos[0])-1 and int(infos[0])-1+1 == b2.lidx) : 
													if b1.forme < b2.forme :
														binome = Binome(b1,b2, b1.cat)
														if infos[1] == "et":
															binomes[binome][0] +=1 #relié par et dans l'ordre alphabétique
														else : 
															binomes[binome][1] +=1  #relié par ou dans l'ordre alphabétique
													else:
														binomeBis = Binome(b2, b1, b1.cat)
														if infos[1] == "et":
															binomes[binomeBis][2] += 1 #relié par et dans l'ordre non alphabétique
														else :
															binomes[binomeBis][3] +=1 #relié par ou dans l'ordre non alphabétique

												# BINOME INDIRECT
												else : 
													if b1.forme < b2.forme :
														binome = Binome(b1,b2, b1.cat)
														if infos[1] == "et":
															binomes[binome][4] +=1 #relié par et dans l'ordre alphabétique
														else : 
															binomes[binome][5] +=1  #relié par ou dans l'ordre alphabétique
													else:
														binomeBis = Binome(b2, b1, b1.cat)
														if infos[1] == "et":
															binomes[binomeBis][6] += 1 #relié par et dans l'ordre non alphabétique
														else :
															binomes[binomeBis][7] +=1 #relié par ou dans l'ordre non alphabétique
												
												break #on s'arrête dès qu'on a trouvé un gouverneur

									except ValueError:
										break
						except ValueError :
							break

				sentence = [] #on réinitialise la phrase

def extractLineaire(filename):
	''' Extrait linéairement les binômes présents dans le fichier source et les enregistre dans le dictionnaire "binomes"

	*INPUT :
		sourceFilename (String) _ un nom de fichier conll avec des phrases
	*OUTPUT :
		void _ remplit le dictionnaire "binomes"
	'''
	with open (filename,"r") as stream :
		#print(filename)

		sentence = [] #liste qui contient à tour de rôle les phrases

		for line in stream :
			infos=line.strip().split()
			if(infos != []):
				sentence.append(infos) #on les ajoute à la phrase tant qu'on a pas de ligne vide
			
			else : #on est sur une ligne vide => fin de la phrase
				
				for i in range(len(sentence)-1) : #pour chacun des mots

					if (sentence[i][1] == "et" or sentence[i][1] == "ou") and (i>0) : #si on est sur un coordinateur pas aux extrémités
						
						##BINOME DIRECT
						if(isBinome(sentence[i-1], sentence[i+1])):

							b1 = Element(sentence[i-1])
							b2 = Element(sentence[i+1])					
							if b1.forme < b2.forme :
								binome = Binome(b1,b2,b1.cat)
								if sentence[i][1] == "et":
									binomes[binome][0] +=1 #relié par et dans l'ordre alphabétique
								else : 
									binomes[binome][1] +=1  #relié par ou dans l'ordre alphabétique
							else:
								binome = Binome(b2,b1,b1.cat)
								if sentence[i][1] == "et":
									binomes[binome][2] += 1 #relié par et dans l'ordre non alphabétique
								else : 		
									binomes[binome][3] +=1 #relié par ou dans l'ordre non alphabétique
						
						##BINOME INDIRECT
						if((i>1 and i<len(sentence)-2) and ((sentence[i+1][3] == "DET" and sentence[i-2][3] == "DET") or (sentence[i+1][3] == "P" and sentence[i-2][3] == "P")) ):
							
							if(isBinome( sentence[i-2], sentence[i+1])):

								b1 = Element(sentence[i-1])
								b2 = Element(sentence[i+2])
								if(b1.forme<b2.forme):
									binome = Binome(b1,b2,b1.cat)
									if sentence[i][1] == "et":
										binomes[binome][4] +=1 #relié par et dans l'ordre alphabétique
									else : 
										binomes[binome][5] +=1  #relié par ou dans l'ordre alphabétique
								else:
									binome = Binome(b2,b1,b1.cat)
									if sentence[i][1] == "et":
										binomes[binome][6] += 1 #relié par et dans l'ordre non alphabétique
									else : 		
										binomes[binome][7] +=1 #relié par ou dans l'ordre non alphabétique

				sentence = [] #on réinitialise la phrase#EXTRACTION LINEAIRE

def saveBinoms(saveFilename,dico):
	''' Enregistre les binômes du dictionnaire dico dans le fichier saveFileName
	*INPUT : 
		saveFilename (String) _ le nom du fichier dans lequel on enregistre les binômes
		dico (defaultdict(int)) _ le dictionnaire dans lequel sont stockés les binômes à enregistrer
	*OUTPUT :
		void _ Enregistre les binômes dans un fichier txt
	'''
	with open(args.methode+"_"+saveFilename+".txt", "w") as stream:
		print(len(dico)," binômes ont été extraits.")
		for binome in dico :
			stream.write(str(binome.classe)+" "+str(binome)+" "+str(dico[binome]).strip("]").strip('[')+"\n")# ENREGISTRE L'EXTRACTION DANS UN .txt

def assignClass(dico):

	# Les classes assignées vont de 1 à 10
	# 10 - figé dans l'ordre alphabétique
	# jusqu'à 
	# 1 - figé dans l'ordre inverse

	for binome in dico.keys():
		binome.totalOcc = sum(x for x in dico[binome])
		ordre_alpha = dico[binome][0]+dico[binome][1]+dico[binome][4]+dico[binome][5]
		ordre_inverse = binome.totalOcc-ordre_alpha

		prop_alpha = (ordre_alpha/binome.totalOcc) * 100
		if(prop_alpha <= 10) :
			binome.classe=1
		elif(prop_alpha<=20) :
			binome.classe=2
		elif(prop_alpha<=30) :
			binome.classe=3
		elif(prop_alpha<=40) :
			binome.classe=4
		elif(prop_alpha<=50) :
			binome.classe=5
		elif(prop_alpha<=60) :
			binome.classe=6
		elif(prop_alpha<=70) :
			binome.classe=7
		elif(prop_alpha<=80) :
			binome.classe=8
		elif(prop_alpha<=90) :
			binome.classe=9
		elif(prop_alpha<=100) :
			binome.classe=10

		
#----------------------------------------#
#----------------MAIN--------------------#
#----------------------------------------#

#binomes (Dico) _ un dictionnaire contenant les binomes extraits
		#	*clé (binôme) : 	(m1, m2)
		#	*val (liste) : 	[a, b, c, d, e, f, g, h]
		#					a = nombre de binômes directs reliés par et dans l'ordre alpha 
		#					b = nombre de binômes directs reliés par ou dans l'ordre alpha
		#					c = nombre de binômes directs reliés par et dans l'ordre inverse 
		#					d = nombre de binômes directs reliés par ou dans l'ordre inverse
		#					e = nombre de binômes indirects reliés par et dans l'ordre alpha 
		#					f = nombre de binômes indirects reliés par ou dans l'ordre alpha
		#					g = nombre de binômes indirects reliés par et dans l'ordre inverse 
		#					h = nombre de binômes indirects reliés par ou dans l'ordre inverse 

binomes = defaultdict(lambda : [0 for x in range(0,8)]) 

### 1. EXTRACTION

def listdirectory(path) : 
	fichier = []
	for root, dirs, files in os.walk(path) : 
		for x in files :
			fichier.append(os.path.join(root,x))
	return fichier

if(args.methode == "dependance"): # Extraction par dépendances
	for file in listdirectory(args.rootd) :
		extractDependance(file)

elif(args.methode == "lineaire") : #Extraction linéaire
	for file in listdirectory(args.rootd) :
		extractLineaire(file)
else :
	print("la méthode choisie n'est pas correcte")
	exit()

# 2. ASSIGNER LES CLASSES

assignClass(binomes)

# 3. ENREGISTREMENT 

saveBinoms(args.save_file, binomes) #Dans le fichier txt

pickle.dump(dict(binomes), open(args.save_file+".pkl", "wb")) #Dans le fichier pkl
#NB : besoin d'un cast pour le dico car pickle ne prend pas les fonctions lambda