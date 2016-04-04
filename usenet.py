import json
import os.path
import string
import numpy as np

import time

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import word_tokenize
from nltk.corpus import stopwords        
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from text_similarity import most_similar, cos_similarity

from scipy.sparse import csr_matrix, vstack

def kmeans(population, k=5):
	estimator = KMeans(n_clusters=k)
	estimator.fit(population)
	return estimator.labels_

def preprocessing():
	categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
	twenty_train = fetch_20newsgroups(categories=categories)

	json_data = {"query": twenty_train.data[-1]}
	json_data["corpus"] = twenty_train.data[:-1]

	print(json_data['query'])

	with open('files/tarea03.json', 'w') as outfile:
		json.dump(json_data, outfile)

def task03a(k=6, N=10):
	t = time.time()
	path = 'files/tarea03.json'
	if not os.path.exists(path):
		preprocessing()

	with open(path, 'r') as outfile:
		json_data = json.load(outfile)

	corpus = json_data['corpus']
	query = json_data['query']

	# NLTK languages stop words. to prevent stop words from tfidf analysis.
	stop = stopwords.words('english')

	# NLTK stemming. Words are transformed to its root.
	stemmer = SnowballStemmer('english', ignore_stopwords=True)

	def stem_tokens(tokens, stemmer):
		stemmed = []
		for item in tokens:
			stemmed.append(stemmer.stem(item))
		return stemmed

	def tokenize(text):
		# Prevent punctuations ,.;... to occur in words
		text = "".join([ch for ch in text if ch not in string.punctuation])
		tokens = word_tokenize(text)
		stems = stem_tokens(tokens, stemmer)
		return stems

	vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=stop)
	tfidf = csr_matrix(vectorizer.fit_transform(corpus).toarray())

	q = csr_matrix(vectorizer.transform([query]).toarray()[0]) # query weight vector

	population = vstack([tfidf, q])	

	labels = kmeans(population, k)
	label = labels[-1]
	labels = labels[:-1]

	new_corpus = np.array(corpus)
	new_corpus = new_corpus[np.where(labels==label)[0]]
	
	tfidf = vectorizer.fit_transform(new_corpus).toarray()
	q = vectorizer.transform([query]).toarray()[0]

	similarities = np.zeros(tfidf.shape[0])
	for i in range(tfidf.shape[0]):
		similarities[i] = cos_similarity(tfidf[i],q)
	
	indexes = similarities.argsort()[-N:][::-1] # Get indexes of N max values. K most similar documents
	print(indexes)
	print(time.time()-t)



def task03b(N=10):
	t = time.time()
	path = 'files/tarea03.json'
	if not os.path.exists(path):
		preprocessing()

	with open(path, 'r') as outfile:
		json_data = json.load(outfile)

	corpus = json_data['corpus']
	query = json_data['query']

	similar = most_similar(corpus, query, N=N, language='english')

	print(similar)
	print(time.time()-t)

if __name__=='__main__':
	task03b()