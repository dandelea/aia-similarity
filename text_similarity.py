import json
import numpy as np
import string

from StemTokenizer import StemTokenizer

from nltk import word_tokenize
from nltk.corpus import stopwords        
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

def most_similar(vectorizer, query, N, language):
	tfidf = vectorizer.toarray()

	q = vectorizer.transform([query]).toarray()[0] # query weight vector

	similarities = np.zeros(tfidf.shape[0])
	for i in range(tfidf.shape[0]):
		similarities[i] = cos_similarity(tfidf[i],q)
	
	indexes = similarities.argsort()[-N:][::-1] # Get indexes of N max values. K most similar documents
	return indexes

def cos_similarity(v, w):
	return np.dot(v,w) / (np.linalg.norm(v)*np.linalg.norm(w))

def task02(N=5):
	# Read corpus and query json
	with open('files/tarea02.json', 'r') as outfile:
		json_data = json.load(outfile)

	query = json_data["query"]
	corpus = json_data["corpus"]

	# NLTK languages stop words. to prevent stop words from tfidf analysis.
	stop = stopwords.words('spanish')

	vectorizer = TfidfVectorizer(tokenizer=StemTokenizer('spanish'), stop_words=stop)
	vectorizer.fit_transform(corpus)

	result = most_similar(vectorizer, query, N, 'spanish')
	print(result)

if __name__=='__main__':
	task02()