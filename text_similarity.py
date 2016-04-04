import json
import numpy as np
import string

from nltk import word_tokenize
from nltk.corpus import stopwords        
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

def most_similar(corpus, query, N=5, language='english'):

	# NLTK languages stop words. to prevent stop words from tfidf analysis.
	stop = stopwords.words(language)

	# NLTK stemming. Words are transformed to its root.
	stemmer = SnowballStemmer(language, ignore_stopwords=True)

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
	tfidf = vectorizer.fit_transform(corpus).toarray()

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

	a = most_similar(corpus, query, N, language='spanish')
	print(a)

if __name__=='__main__':
	task02()