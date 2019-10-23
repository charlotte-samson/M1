#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import bz2
import argparse
from collections import defaultdict
import pickle
usage_str=u""" Ce programme lit un fichier bz2 contenant des vecteurs
"""


argparser = argparse.ArgumentParser(usage = usage_str)
"""argparser.add_argument('dev', help='fichier annoté au format CoNLL,corpus dev')
argparser.add_argument('train', help='fichier annoté au format CoNLL, corpus train')
argparser.add_argument('test', help='fichier annoté au format CoNLL, corpus test')"""
argparser.add_argument('vecs', help='fichier avec les vecteurs correspondant aux mots au format bz2')
args = argparser.parse_args()


#créé un dictionnaire mot --> vec, vec étant une liste de float à partir d'un fichier bz2 
word2vec=defaultdict(list)
with bz2.open (args.vecs,"rt") as flow : 
    for line in flow :
        vec=line.strip().split()
        word2vec[vec[0]]=[float(i) for i in vec[1:]]

print("sous quel nom enregistrer les embeddings ?")
filename=input()
pickle.dump(word2vec, open(filename, "wb" ) )

