###########################
# Projet semestriel PROBA #
###########################

#----------Contenu de l'archive-----------------
- objets.py
- extraction.py
- affiche.py
- classifieur.py
- README.txt

#----------Utilisation du code-----------------

# objets.py
Code contenant uniquement les deux objets dont nous nous servons dans ce projet
A savoir : Element et Binome
Ce fichier ne s'execute pas il factorise simplement le code necessaire dans les 3 autres.

# extraction.py
Code contenant les fonctions nécessaires à l'extraction des binômes.
Il prend en ligne de commande 3 arguments : 
	- un chemin vers un dossier contenant des fichiers conll ou des dossiers qui en contiennent eux -mêmes 
	(ceux desquels il faut extraire les binomes)
	- le nom d'un fichier dans lequel enregistrer les binômes extraits
	- le nom d'une méthode ("lineaire" ou "dependance")

ligne de commande : python3 extraction.py /DOSSIER/ saveFile methode

# affichage.py
Code contenant les méthodes qui affichent les graphiques qui analysent les données.
Il prend en ligne de commande 2 arguments :
	- le nom d'un fichier .pkl contenant un dictionnaire de Binômes
	(celui-ci doit avoir la forme donnée par le fichier extraction.py)
	- le nom d'un fichier pour la sauvegarde du dictionnaire contenant les nouveaux binômes nettoyés

ligne de commande : python3 affiche.py dictionnaire_binomes.pkl saveFile

# classfieur.py
Code contenant le perceptron qui classifie les binômes extraits.
Il prend en ligne de commande 3 arguments :
	- le nom d'un fichier .pkl contenant un dictionnaire de Binômes.
	(celui-ci doit avoir la forme donnée par extraction.py ou affiche.py)
	- le nom d'un fichier .pkl d'annotations sémantiques (dictionnaire)
	- le nom d'un fichier .json d'annotations phonologiques 

ligne de commande : python3 classifieur.py dictionnaire_binomes.pkl annotations_semantiques.pkl annotations_phonologiques.json

#-------------Auteurs---------------------
Projet réalisé par Charlotte DUGRAIN Alexandra GUÉRIN.
