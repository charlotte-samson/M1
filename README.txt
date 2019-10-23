Auteurs: 
Charlotte Dugrain, William Durand, Vincent Rivier

Fichiers présents dans le rendu:
- ExtractData.py
- extractVec.py
- main.py
- matplot.py
- NN_current.py
- POS_MFTagger_baseline.py
- embeddings
- train_lemme
- train_words
- RapportDurandRivierDugrain.pdf

Manuel d’utilisateur : 

si aucun fichier pickle ‘embeddings’ n’est pas fourni:

executer extractVec.py suivi du nom du fichier contenant les embeddings pré entraînés (vecs50-linear-frwiki.bz2)
executer main.py en ajoutant le fichier pickle, le fichier connl du corpus train suivi de celui du dev et enfin celui du test

Si un fichier pickle ‘embeddings’ est fourni : 

executer main.py en ajoutant le fichier pickle, le fichier connl du corpus train suivi de celui du dev et enfin celui du test
ex : python3 main.py embeddings fr-ud-train.conllu fr-ud-dev.conllu fr-ud-test.conllu

Vous serez ensuite guidé, pour savoir si vous souhaitez au choix : 

entraîner le réseau puis le tester sur le corpus test
tester sur le corpus test un réseau déjà entraîné enregistré sous forme de fichier pickle
tester sur une phrase un réseau entraîné enregistré  sous forme de fichier pickle

