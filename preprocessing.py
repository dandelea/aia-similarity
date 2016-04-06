import json
import numpy as np
import os.path
import pickle

from nltk.corpus import stopwords

from StemTokenizer import StemTokenizer

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def preprocessing(task_name, language='english', k=5):

	documents_filepath = 'files/' + task_name + '.json'
	output_filepath = 'files/' + task_name + '-' + str(k) + '.pkl'

	if os.path.exists(output_filepath):
		os.remove(output_filepath)

	if not os.path.exists(documents_filepath):
		categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
		usenet_data = fetch_20newsgroups(categories=categories).data
		with open(documents_filepath, 'w') as outfile:
			json.dump(usenet_data, outfile)
	else:
		with open(documents_filepath, 'r') as outfile:
			usenet_data = json.load(outfile)

	usenet_data = np.array(usenet_data)

	# NLTK languages stop words.
	# To prevent stop words from tfidf analysis.
	stop = stopwords.words(language)

	vectorizer = TfidfVectorizer(tokenizer=StemTokenizer(language), stop_words=stop)
	population = vectorizer.fit_transform(usenet_data)
	estimator = kmeans(population, k)

	output_data = {
		"vectorizer": vectorizer,
		"tfidf" : population,
		"estimator" : estimator
	}

	with open(output_filepath, 'wb') as pfile:
		pickle.dump(output_data,pfile)

def kmeans(population, k=5):
	estimator = KMeans(n_clusters=k)
	estimator.fit(population)
	return estimator