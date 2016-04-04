import numpy as np
import random
import matplotlib.pyplot as plt

from plot import scatterplot_matrix

from sklearn.datasets import load_iris

def data():
	return load_iris().data

def generateK(population, k):
	sample = random.sample(list(population), k)
	return np.array(sample)
	#return np.array([[5.8,2.7,3.9,1.2], [6.5,3,5.2,2]])

def mean(population, indexes):
	cols = population.shape[1]
	r = np.zeros(cols)
	for i in range(cols):
		r[i] = np.mean(population[indexes, i])
	return r

def kmeans(population, k=3):
	if k>0:

		y = np.zeros(population.shape[0]) # Clasificacion del individuo i-esimo

		groups = generateK(population, k) # Coordenadas iniciales de los grupos

		has_changed = True
		while has_changed:

			# Por cada individuo
			for i in range(population.shape[0]):
				element = population[i]
				distancias = []
				# calcula las distancias con los grupos
				for group in groups:
					distance = np.linalg.norm(element - group)
					distancias.append(distance)
				# clasificacion: Se queda con aquel grupo de distancia minima
				y[i] = np.argmin(distancias)

			# Por cada grupo, recalcula la media de sus caracteristicas
			for i in range(groups.shape[0]):
				indexes = np.where(y==i)
				m = mean(population, indexes)
				has_changed &= not np.array_equal(m, groups[i])
				groups[i] = m

			plot(population, groups)

		return groups
	else:
		raise ValueError('K value must be positive')

def plot(population, groups):

	# Por cada individuo
	y = np.zeros(population.shape[0])
	for i in range(population.shape[0]):
		element = population[i]
		distances = []
		# calcula las distancias con los grupos
		for group in groups:
			distance = np.linalg.norm(element - group)
			distances.append(distance)
		# clasificacion: Se queda con aquel grupo de distancia minima
		y[i] = np.argmin(distances)

	for group in groups:
		population = np.vstack([population, group])
		y = np.append(y,-1)

	scatterplot_matrix(population, y)

	'''for i in range(1,5):
		for j in range(1,5):
			if i!=j:
				plt.subplot(16,i,j)
				plt.scatter(population[:,i-1], population[:,j-1], c=y)
			
	plt.show()'''

if __name__=='__main__':
	population = data()
	aux = kmeans(population, 2)
	print(aux)
