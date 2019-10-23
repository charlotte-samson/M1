#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

	Classe Neural Network implémentant un réseau de neurones de type "feed forward"


	Organisation du code :
	- initialisation du réseau et des couches
	- fonction d'entraînement, propagation avant, propagation arrière
	- fonctions de test


	Notations
	- w réfère aux poids / paramètres connectant une couche à la précédente
	- z réfère aux valeurs des neurones avant activation
	- a réfère aux valeurs des neurones après activation

"""

import numpy as np
import random
import matplotlib.pyplot as plt
from math import log
from math import sqrt



class NeuralNetwork:

	def __init__(self, nb_neurons, activation_function, output_function, loss_function):

		self.layers = list()

		# input layer
		self.layers.append(self.Layer(0, 0, None))
		# hidden layers
		self.layers.extend([self.Layer(nb_neurons[x], nb_neurons[x - 1], activation_function=activation_function)
							for x in range(1, len(nb_neurons) - 1)])
		# output layer
		self.layers.append(self.Layer(nb_neurons[-1], nb_neurons[-2], output_function))

		self.loss_function = loss_function


	class Layer:

		"""
			Les couches sont constituées de deux sous-couches Z_layer et A_layer (respectivement avant et après activation, conformément aux notations)
			Les vecteurs et les paramètres sont stockés dans les sous-couches, les gradients sont stockés dans la couche elle-même

			On distingue 3 types de couches différentes:
			- couche d'input, qui se définit uniquement par ses valeurs de sortie ou a_vector. Cette valeur est passée depuis l'extérieur du réseau.
			- couche cachée, connectées avec les poids (weights), avec des valeurs et une fonction d'activation pour la passe avant, et des valeurs pour la passe arrière
			- couche d'output, qui est une couche cachée avec une fonction d'activation propre

		"""
	
		def __init__(self, nb_neurons, prev_nb_neurons, activation_function):
			self.prev_nb_neurons = prev_nb_neurons
			self.nb_neurons = nb_neurons
			self.z_layer = self.Z_layer(nb_neurons, prev_nb_neurons)
			self.a_layer = self.A_layer(activation_function)
			self.update_matrix = 0
			self.dE_dz = 0

		def reset_update(self):
			self.update_matrix = 0
			self.dE_dz = 0

		class Z_layer:
			def __init__(self, nb_neurons, prev_nb_neurons):
				if prev_nb_neurons + nb_neurons > 0:
					np.random.seed(1)
					self.weights = np.random.uniform(-(sqrt(6) / sqrt(prev_nb_neurons + nb_neurons)),
													 +(sqrt(6) / sqrt(prev_nb_neurons + nb_neurons)),
													 [prev_nb_neurons, nb_neurons])
					# Xavier initialization
				self.biases = np.random.rand(nb_neurons, )
				self.z_vector = 0

		class A_layer:
			def __init__(self, activation_function):
				self.activation_function = activation_function
				self.a_vector = 0


	# - - - - - - - - - - - - - - - - - - - - - - - -


	# Fonction d'entraînement du réseau
	# appelle successivement les fonctions feed_forward > backpropagation > update
	def train(self, X_train, y_train, epoch=1, learning_rate=0.5, mini_batch=1):
		last_tag_index = 0
		for e in range(epoch):
			for i in range(0, len(X_train)):
				X_train[i, 0:18] = 0
				X_train[i, last_tag_index] = 1
				last_tag_index = np.argmax(self.feed_forward(X_train[i]))
				self.backpropagation(y_train[i])

				if i % mini_batch == 0:
					self.update(learning_rate, mini_batch)



	# Propagation avant de l'input

	def feed_forward(self, input_vector):
		self.layers[0].a_layer.a_vector = input_vector

		for i in range(1, len(self.layers)):
			previous_layer = self.layers[i - 1]
			layer = self.layers[i]

			# calcul z_vector
			input_vector = previous_layer.a_layer.a_vector
			weights = layer.z_layer.weights
			biases = layer.z_layer.biases
			layer.z_layer.z_vector = np.dot(input_vector, weights) + biases

			# calcul a_vector
			activation_function = layer.a_layer.activation_function
			layer.a_layer.a_vector = activation_function(layer.z_layer.z_vector)

		output = self.layers[-1].a_layer.a_vector
		return output



	# Propagation arrière

	# 1) itère sur les layers en sens inverse dans backpropagation()
	# 2) calcul des gradients pour chaque layer dans output/hidden_derivation()

	# notations
	# dE_da : dérivée partielle de l'Erreur par rapport à l'activation
	# da_dz : dérivée partielle de l'activation par rapport à la préactivation
	# dz_dw : dérivée partielle de la préactivation par rapport aux poids

	# itération en sens inverse
	def backpropagation(self, y):
		self.output_derivation(y)

		for i in range(2, len(self.layers)):
			self.hidden_derivation(-i)

	# calcul des gradients
	# - les gradients une fois calculés sont stockés en attributs des layers
	# - les dérivées de fonctions sont en fin de code
	def output_derivation(self, y):
		output_layer = self.layers[-1]
		output = output_layer.a_layer.a_vector
		dE_da = mean_squared_derivative(output, y)
		da_dz = sigmoid_derivative(output)
		dz_dw = self.layers[-2].a_layer.a_vector

		output_layer.dE_dz += dE_da * da_dz
		output_layer.update_matrix += np.dot(np.asmatrix(dz_dw).T, np.asmatrix(output_layer.dE_dz))

	def hidden_derivation(self, i):
		hidden_layer = self.layers[i]
		a_vector = hidden_layer.a_layer.a_vector

		dE_da = np.dot(self.layers[i + 1].dE_dz, self.layers[i + 1].z_layer.weights.T)
		da_dz = sigmoid_derivative(a_vector)
		dz_dw = self.layers[i - 1].a_layer.a_vector

		hidden_layer.dE_dz += dE_da * da_dz
		hidden_layer.update_matrix += np.dot(np.asmatrix(dz_dw).T, np.asmatrix(hidden_layer.dE_dz))


	# mise à jour des paramètres
	# appelée dans train()
	def update(self, learning_rate, mini_batch):

		for layer in self.layers[1:]:
			layer.z_layer.weights -= (layer.update_matrix / mini_batch) * learning_rate
			layer.z_layer.biases -= (np.sum(layer.dE_dz / mini_batch)) * learning_rate
			layer.reset_update()

		

	# Tests
	#prediction successive sur tout le corpus passé en arguments, renvoie le pourcentage de bonnes prédictions
	def test(self, X_corp, Y_corp, corpwords, pos_semantics, logs=False):
		last_tag_index = 0
		good_answers = 0
		confusion_matrix = np.zeros((len(pos_semantics),len(pos_semantics)))
		table_confusion=np.zeros((len(pos_semantics),4))
		for i in range(0, len(X_corp)):
			X_corp[i, 0:18] = 0
			X_corp[i, last_tag_index] = 1
			prediction = self.predict(X_corp[i])
			if prediction == np.argmax(Y_corp[i]):
				confusion_matrix[np.argmax(Y_corp[i]),prediction]+=1
				good_answers += 1
			else :
				confusion_matrix[np.argmax(Y_corp[i]),prediction]+=1
			if logs:
				with open("log_file.txt", "w+") as wr:
					wr.write(corpwords[i] + ": " + pos_semantics[prediction])
		#en lever le commentare ci dessous si on souhaite afficher la matrice de confusion et les tables de confusions par classe
		"""
		print(confusion_matrix)
		for i in range(len(pos_semantics)) : 
			table_confusion[i,0]=confusion_matrix[i,i]
			table_confusion[i,1]=sum(confusion_matrix[i+1:,i])+sum(confusion_matrix[0:i-1,i])
			table_confusion[i,2]=sum(confusion_matrix[i,i+1:])+sum(confusion_matrix[i,0:i-1])
			table_confusion[i,3]=sum(sum(confusion_matrix))-(table_confusion[i,0]+table_confusion[i,1]+table_confusion[i,2])
		print(table_confusion)
		"""
		return good_answers / len(X_corp) * 100



	#prédiction
	#renvoie l'indice de l'output du score le plus haut
	def predict(self, input_vector):
		return np.argmax(self.feed_forward(input_vector), axis=0)




# dérivées des fonctions
def sigmoid_derivative(output):
	return output * (1 - output)


def mean_squared_derivative(output, y):
	return output - y


def softmax_derivative(sm):
	s = sm.reshape(-1, 1)
	return np.diagflat(s) - np.dot(s, s.T)


def reLU_derivative(x):
	return 1. * (x > 0)
