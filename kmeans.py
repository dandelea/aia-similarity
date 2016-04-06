import math
import numpy as np
from plot import plot
import random

import time

from sklearn.datasets import load_iris

def iris_data():
	return load_iris().data

def generateK(population, k):
	sample = random.sample(list(population), k)
	return np.array(sample)
	#return np.array([[5.8,2.7,3.9,1.2], [6.5,3,5.2,2]])

def distance_points(p, q):
	vector = p-q
	sum = 0
	for i in vector:
		sum += i**2
	return math.sqrt(sum)

def mean(population, indexes):
	cols = population.shape[1]
	r = np.zeros(cols)
	for i in range(cols):
		r[i] = np.mean(population[indexes, i])
	return r

def kmeans(population, k=3, display_plots=False):
	if k>0:

		y = np.zeros(population.shape[0]) # Clasificacion del individuo i-esimo

		groups = generateK(population, k) # Coordenadas iniciales de los grupos

		if display_plots:
			for file in os.listdir('files'):
				if file.endswith('.png') and file.startswith('iris_'):
					os.remove('files/' + file)

		has_changed = True
		it = 1
		while has_changed:

			# Por cada individuo
			for i in range(population.shape[0]):
				element = population[i]
				distancias = []
				# calcula las distancias con los grupos
				for group in groups:
					distance = distance_points(element, group)
					distancias.append(distance)
				# clasificacion: Se queda con aquel grupo de distancia minima
				y[i] = np.argmin(distancias)

			# Por cada grupo, recalcula la media de sus caracteristicas
			for i in range(groups.shape[0]):
				indexes = np.where(y==i)
				m = mean(population, indexes)
				has_changed &= not np.array_equal(m, groups[i])
				groups[i] = m

			if display_plots:
				plot(population, groups, it)
			it += 1

		return groups
	else:
		raise ValueError('K value must be positive')

def task01(k=5, display_plots=True):
	population = iris_data()
	groups = kmeans(population)
	print(groups, k, display_plots)

def test_task01_time(maxk = 100):
	t = time.time()
	population = iris_data()
	k = 2
	while (k<=maxk):
		t = time.time()
		print("k: " + str(k) + "/" + str(maxk))
		kmeans(population, k)
		print("Tiempo: " + str(time.time() - t) + " s.")
		k+=1

if __name__=='__main__':
	test_task01_time()