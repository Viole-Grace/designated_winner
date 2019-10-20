import io
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import nltk
import random
import pickle

from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def get_features(pos, neg):

	wordlib = []

	for fi in [pos, neg]:
		with io.open(fi, 'r', encoding='cp437') as f:
			content = f.readlines()
			for l in content:
				all_words = word_tokenize(l.lower())
				wordlib+=list(all_words)
	
	wordlib_stem, wordlib_lem=[stemmer.stem(x) for x in wordlib], [lemmatizer.lemmatize(x) for x in wordlib]
	word_count_stem, word_count_lem = Counter(wordlib_stem), Counter(wordlib_lem)

	stem_lib, lem_lib = [],[]

	for w in word_count_stem:
		if 1200>word_count_stem[w]>50:
			stem_lib.append(w)

	for w in word_count_lem:
		if 1200>word_count_lem[w]>50:
			lem_lib.append(w)

	return stem_lib, lem_lib

def feature_embedding_lem(sample, wordlib, label):

	feature_set = []

	with io.open(sample, 'r', encoding='cp437') as f:
		content = f.readlines()
		for l in content:
			curr_words = word_tokenize(l.lower())
			curr_words = [lemmatizer.lemmatize(x) for x in curr_words]
			features = np.zeros(len(wordlib))

			for word in curr_words:
				if word.lower() in wordlib:
					features[wordlib.index(word.lower())]+=1
			features = list(features)
			feature_set.append([features, label])

	return feature_set

def feature_embedding_stem(sample, wordlib, label):

	feature_set = []

	with io.open(sample, 'r', encoding='cp437') as f:
		content = f.readlines()
		for l in content:
			curr_words = word_tokenize(l.lower())
			curr_words = [stemmer.stem(x) for x in curr_words]
			features = np.zeros(len(wordlib))

			for word in curr_words:
				if word.lower() in wordlib:
					features[wordlib.index(word.lower())]+=1
			features = list(features)
			feature_set.append([features, label])

	return feature_set

def create_features_and_labels(pos, neg, test_size=0.1):

	wordlib_stem, wordlib_lem = get_features(pos,neg)
	features_stem, features_lem = [],[]

	#stemmed words
	features_stem += feature_embedding_stem('datasets/SentimentAnalysis/pos.txt', wordlib_stem, 1)
	features_stem += feature_embedding_stem('datasets/SentimentAnalysis/neg.txt', wordlib_stem, 0)
	random.shuffle(features_stem)

	#convert to np array for training / testing
	features_stem = np.array(features_stem)

	#lemmatized words
	features_lem += feature_embedding_lem('datasets/SentimentAnalysis/pos.txt', wordlib_lem, 1)
	features_lem += feature_embedding_lem('datasets/SentimentAnalysis/neg.txt', wordlib_lem, 0)
	random.shuffle(features_lem)

	#cover to np array to training/testing
	features_lem = np.array(features_lem)

	testing_size_lem = int(test_size*len(features_lem))
	testing_size_stem = int(test_size*len(features_stem))

	print " Testing Size (lemmatized): ",testing_size_lem
	print " Testing Size (stemmed): ",testing_size_stem

	print "Features (lemmatized) : ",features_lem[0]
	print "Features (stemmed) : ",features_stem[0]

	print "Number of features (lemmatized) : ",len(features_lem[0][0])
	print "Number of features (stemmed) : ", len(features_stem[0][0])

	#create x_train and x_test, y_train and y_test for both lemmatized words and stemmed words 
	x_lem_tr = list(features_lem[:,0][:-testing_size_lem])
	y_lem_tr = list(features_lem[:,1][:-testing_size_lem])
	x_lem_te = list(features_lem[:,0][-testing_size_lem:])
	y_lem_te = list(features_lem[:,1][-testing_size_lem:])

	x_stem_tr = list(features_stem[:,0][:-testing_size_stem])
	y_stem_tr = list(features_stem[:,1][:-testing_size_stem])
	x_stem_te = list(features_stem[:,0][-testing_size_stem:])
	y_stem_te = list(features_stem[:,1][-testing_size_stem:])

	return x_lem_tr, x_lem_te, y_lem_tr, y_lem_te, x_stem_tr, x_stem_te, y_stem_tr, y_stem_te

def split_and_train():

	pos_filename, neg_filename = 'datasets/SentimentAnalysis/pos.txt','datasets/SentimentAnalysis/neg.txt'

	x_lem_tr, x_lem_te, y_lem_tr, y_lem_te, x_stem_tr, x_stem_te, y_stem_tr, y_stem_te = create_features_and_labels(pos_filename, neg_filename)

	#keep it all in a single set. Format : X_tr, X_te, Y_tr, Y_te
	with open('sentiment_set_lem.pickle','wb') as f:
		pickle.dump([x_lem_tr, x_lem_te, y_lem_tr, y_lem_te], f)

	#keep it all in a single set. Format : X_tr, X_te, Y_tr, Y_te
	with open('sentiment_set_stem.pickle','wb') as f:
		pickle.dump([x_stem_tr, x_stem_te, y_stem_tr, y_stem_te], f)

split_and_train()