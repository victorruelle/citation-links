import csv
from collections import defaultdict
from gensim import corpora, models, similarities
#activate log 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#here we open and process the testing set, ie pairs of articles without a link
with open("Data/testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)
	
testing_set = [element[0].split(" ") for element in testing_set]
#we obtain a list of lists of 2 strings

#here we pre-process the training data
with open("Data/training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]
#we obtain a list of lists of 3 strings: the two articles IDs and 1 if there is a link, 0 otherwise

with open("Data/node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)


# create a list with list with all the words for each info
corpus_abstract = [element[5].split(" ") for element in node_info]
corpus_title = [element[2].split(" ") for element in node_info]
#take out the words that only appear once
frequency1,frequency2 = defaultdict(int),defaultdict(int)
for text in corpus_abstract:
	for token in text:
		frequency1[token] += 1

for text in corpus_title:
	for token in text:
		frequency2[token] += 1

corpus_abstract = [[token for token in text if frequency1[token] > 1]
		for text in corpus_abstract]
corpus_title = [[token for token in text if frequency2[token] > 1]
		for text in corpus_title]
		
#create a dictionary, meaning a specific id for every word 
dictionary1 = corpora.Dictionary(corpus_abstract)
dictionary2 = corpora.Dictionary(corpus_title)

#here we have a bag-of-words representation, corpus_abstract is a list of lists of words, dictionary is a unique id for every word and bow is a list of tuple reprensenting the nb of occurences of one word represented by its ID in the corpus_abstract
corpus_abstract = [dictionary1.doc2bow(text) for text in corpus_abstract]
corpus_title = [dictionary2.doc2bow(text) for text in corpus_title]

#here we create a tf-idf model
tfidf1 = models.TfidfModel(corpus_abstract)
tfidf2 = models.TfidfModel(corpus_title)
tf_idf1 = tfidf1[corpus_abstract]
tf_idf2 = tfidf2[corpus_title]

#now we convert to LSI model
#chose a number of topics, randomly set to 2
#if we change number of topics, we get a more precise similarity measure (not comparing them to two topics but to more)
lsi1 = models.LsiModel(tf_idf1, id2word=dictionary1, num_topics=2)
lsi2 = models.LsiModel(tf_idf2, id2word=dictionary2, num_topics=2)
corpus_abstract_lsi = lsi1[tf_idf1]
corpus_titles_lsi = lsi2[tf_idf2]

#transorm corpus_abstract to LSI space and index it
index1 = similarities.MatrixSimilarity(corpus_abstract_lsi)
index2 = similarities.MatrixSimilarity(corpus_titles_lsi)

#here we compute the similarity between all the documents and the first document
sims1 = index1[corpus_abstract_lsi]
sims2 = index2[corpus_titles_lsi]
#list(enumerate(sims1 or sims2)) gives a list of tuples (index of doc, array[(index of doc, similarity)])
#list(enumerate(sims1 or sims2))[i][1] is the array of similarities for doc i 

