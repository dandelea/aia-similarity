import numpy as np
import matplotlib.pyplot as plt
import os

def plot(population, groups, iteration):

	population_c = np.copy(population)

	# Por cada individuo
	y = np.zeros(population_c.shape[0])
	for i in range(population_c.shape[0]):
		element = population_c[i]
		distances = []
		# calcula las distancias con los grupos
		for group in groups:
			distance = np.linalg.norm(element - group)
			distances.append(distance)
		# clasificacion: Se queda con aquel grupo de distancia minima
		y[i] = np.argmin(distances)

	for group in groups:
		population_c = np.vstack([population_c, group])
		y = np.append(y,-1)

	cont = 1
	for i in range(4):
		for j in range(4):
			plt.subplot(4,4,cont)
			if i==3:
				if j==0:
					plt.xlabel('Sepal Length')
				elif j==1:
					plt.xlabel('Sepal Width')
				elif j==2:
					plt.xlabel('Petal Length')
				elif j==3:
					plt.xlabel('Petal Width')
			if j==0:
				if i==0:
					plt.ylabel('Sepal Length')
				elif i==1:
					plt.ylabel('Sepal Width')
				elif i==2:
					plt.ylabel('Petal Length')
				elif i==3:
					plt.ylabel('Petal Width')
			if i!=j:
				print((i,j))
				plt.scatter(population_c[:,i], population_c[:,j], c=y)
			else:
				plt.xticks([]), plt.yticks([])				
			cont += 1
	plt.savefig('files/iris_'+ str(iteration) +'.png')
	plt.clf()