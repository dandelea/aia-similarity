from nltk import word_tokenize          
from nltk.stem.snowball import SnowballStemmer 

import string

class StemTokenizer(object):
	def __init__(self, language):
		self.stemmer = SnowballStemmer(language, ignore_stopwords=True)
	def __call__(self, text):
		# Prevent punctuations ,.;... to occur in words
		text = "".join([ch for ch in text if ch not in string.punctuation])
		tokens = word_tokenize(text)
		stems = self.stem_tokens(tokens)
		return stems
	def stem_tokens(self, tokens):
		stemmed = []
		for item in tokens:
			stemmed.append(self.stemmer.stem(item))
		return stemmed