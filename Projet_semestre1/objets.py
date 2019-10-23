#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string

#-------------CLASSE ELEMENT-------------
class Element(object) :
	"""Classe représentant un élément d'un binôme
		lidx : l'index linéaire du mot dans la phrase
		forme : la forme du noeud
		cat : la catégorie grammaticale du mot
		lidxgouv : l'index linéaire de son gouverneur

	"""
	def __init__(self, infos):
		'''
			*INPUT :
				infos (list) _ liste des infos sur une ligne de fichier conll
		'''

		self.lidx = int(infos[0])-1 #/!\ on fait -1 pour que l'index corresponde aux indices dans une liste représentant la phrase
		try :
			if (len(infos[1])>2) :
				if(has_endpunctuation(infos[1])) :
					infos[1]=remove_endpunctuation(infos[1])
		except IndexError :
			print(indexerror)
		self.forme = infos[1]
		self.cat = infos[3] 
		for x in infos[5].split("|") :
			if x[0] == 'n' :
				self.nb = x[2]
			if x[0] == 'g' : 
				self.genre = x[2]
		try:
			self.lidxgouv = [int(x)-1 for x in infos[6].split("|")] #/!\ on fait -1 pour que l'index corresponde aux indices dans une liste représentant la phrase
		except ValueError:
			print()

	def __repr__(self):
		return self.forme

	def __eq__(self,element):
		return (self.forme == element.forme) and (self.cat == element.cat)

	def __hash__(self):
		return hash((self.forme))

#--------------CLASSE BINOME-------------
class Binome(object):
	"""Classe représentant un binome
		elt1 (Element) : le premier élément du binôme
		elt2 (Element) : le deuxième élément du binôme
		coord (String) : "et" ou "ou"

		totalOcc (int) : le nombre total d'apparitions du binôme dans le corpus
		classe (int) : la classe assignée au binôme après avoir réalisé les comptes
	"""

	def __init__(self, elt1, elt2, cat):
		self.elt1 = elt1
		self.elt2 = elt2
		self.cat = cat

		self.totalOcc = None
		self.classe = None

	def __repr__(self):
		return str(self.elt1)+" "+str(self.elt2)+" "+self.cat

	def __eq__(self, binome):
		return (self.elt1 == binome.elt1) and (self.elt2 == binome.elt2) and (self.cat == binome.cat)

	def __hash__(self):
		return hash((self.elt1,self.elt2,self.cat))

	def write(self, saveFilename):
		""" Ecrit le binôme à la suite du fichier saveFilename
			*INPUT : 
				saveFileName(String) : le nom du fichier dans lequel enregistrer les binômes
		"""
		with open (saveFilename, "a") as fichier : #le mode "a" (append) permet d'ajouter à la fin du fichier, le créé s'il existe pas
			fichier.write("\n"+str(self.elt1.lidx)+"\t"+self.elt1.forme+"\t"+self.elt1.cat+"\t"+str(self.elt1.lidxgouv))
			fichier.write("\n"+str(self.elt2.lidx)+"\t"+self.elt2.forme+"\t"+self.elt2.cat+"\t"+str(self.elt2.lidxgouv))
			fichier.write("\n\n")


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