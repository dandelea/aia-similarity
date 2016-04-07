import numpy as np
import pickle

from preprocessing import preprocessing 
from text_similarity import cos_similarity

def task03_pre(k):
	preprocessing('task03', 'english', k)

def task03_run(query, k, N):
	filepath = 'files/task03-' + str(k) + '.pkl'
	with open(filepath, 'rb') as pfile:
		data = pickle.load(pfile)

	vectorizer = data['vectorizer']
	population = data['tfidf']
	estimator = data['estimator']

	q = vectorizer.transform([query]).toarray()

	q_group = estimator.predict(q)[0]

	q = q[0]

	candidates = np.where(estimator.labels_==q_group)[0]

	population = population[candidates]

	similarities = np.zeros(population.shape[0])
	for i in range(population.shape[0]):
		row = population[i].toarray()
		similarities[i] = cos_similarity(row,q)
	
	indexes = similarities.argsort()[-N:][::-1] # Get indexes of N max values. K most similar documents
	indexes = [candidates[ind] for ind in indexes]
	print(indexes)