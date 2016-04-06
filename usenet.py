import json

import numpy as np
import pickle
import random

import time

from preprocessing import preprocessing 
from text_similarity import cos_similarity

from sklearn.datasets import fetch_20newsgroups

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
	return indexes

def test_task03(k=8, N=10):
	#task03_pre(k)

	categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
	usenet = fetch_20newsgroups(categories=categories)

	index = random.randrange(0, len(usenet.data))
	print(index)

	query = usenet.data[index]
	q_family = usenet.target[index]
	print(usenet.target_names[q_family])
	print("--")
	
	most_similar = task03_run(query, k, N)
	
	for similar in most_similar:
		print(usenet.target_names[usenet.target[similar]])



if __name__=='__main__':
	#task03_pre(8)
	#task03_run("Le envio este email porque", k=8, N=10)
	test_task03()