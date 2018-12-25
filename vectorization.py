import csv
from collections import defaultdict
from gensim import corpora, models, similarities
from sklearn.linear_model import LogisticRegression
#activate log 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


'''
THIS FILE CONTAINS METHODS TO COMPUTE DESCRIPTORS FOR OUR DATA

THE MAIN METHODS ARE
	1) features_all(data_set,metas) : takes a data_set = list of (id1,id2) and computes the entire X feature vector
  	SHOULD INCLUDE ALL THE VECTORIZATION METHODS THAT WE WANT TO USE TOGETHER. ONLY ZUCHET'S FEATURES NEED TO BE ADDED.
	2) features(id1,id2,metas) : does the same but for a single link (much to slow because of data passes in metas)

OTHER 'SUPPORT' METHODS:
	1) comput_similarities : Jo's code, computes sims between titles and abstracts (long)
	2) save_sims : save similarity matrices from Jo's code. DO NOT USE as it as (10Go files...).
		WILL ERASE PREVIOUS SAVES WHEN CALLED!
	3) recover_list_sims : recover saved sims 
	4) get_graph_features : recover saved graph features (Zuc.)
'''

def features_all(data_set,metas):
	# must return the vector representation of the all the links (id1,id2) in data_set
	# id1 and id2 are relative to the indexation in node_info !
	# metas contain all the information lists 
	IDs,list_sims1,list_sims2,years,authors = metas


	features = []

	for id1,id2 in data_set:
		x = []

		# converting to the indexing used in the feature lists
		id1 = IDs.index(id1)
		id2 = IDs.index(id2)

		# adding the similarities between titles and descriptions
		x.append(list_sims1[id1][1][id2])
		x.append(list_sims2[id1][1][id2])

		# adding years difference
		x.append(abs(years[id1]-years[id2]))

		# adding number of common authors ( we can learn author habits )
		a1 = authors[id1]
		a2 = authors[id2]
		n = len(set(a1).intersection(set(a2)))
		x.append(n)

		features.append(x)


	return features

def features(id1,id2,metas):
	# MUCH TOO SLOW! list_sims1 and list_sims2 in metas are much too big to be passed 
	# aroun for a single link featuer
	# must return the vector representation of the link (id1,id2)
	# id1 and id2 are relative to the indexation in node_info !
	# metas contain all the information lists 
	IDs,list_sims1,list_sims2,years,authors = metas
	features = []

	# converting to the indexing used in the feature lists
	id1 = IDs.index(id1)
	id2 = IDs.index(id2)

	# adding the similarities between titles and descriptions
	features.append(list_sims1[id1][1][id2])
	features.append(list_sims2[id1][1][id2])

	# adding years difference
	features.append(abs(years[id1]-years[id2]))
	
	# adding number of common authors ( we can learn author habits )
	a1 = authors[id1]
	a2 = authors[id2]
	n = len(set(a1).intersection(set(a2)))
	features.append(n)

	return features



def compute_similarities(corpus_abstract,corpus_title):
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

	list_sims1 = list(enumerate(sims1))
	list_sims2 = list(enumerate(sims2))
	#list(enumerate(sims1 or sims2)) gives a list of tuples (index of doc, array[(index of doc, similarity)])
	#list(enumerate(sims1 or sims2))[i][1] is the array of similarities for doc i 

	return list_sims1,list_sims2
	#save_sims(list_sims1,list_sims2)

def save_sims(list_sims1,list_sims2):
	with open("Data/Processed/list_sims1.dat","w") as sims1:
		for line in list_sims1:
			l = str(line[0])
			for el in line[1]:
				l += " "+str(el)
			l += "\n"
			sims1.write(l)

	with open("Data/Processed/list_sims2.dat","w") as sims2:
		for line in list_sims2:
			l = str(line[0])
			for el in line[1]:
				l += " "+str(el)
			l += "\n"
			sims2.write(l)
			
def recover_list_sims():
	list_sims1,list_sims2 = [],[]
	with open("Data/Processed/list_sims1.dat",'r') as sims1:
		for line in sims1:
			line = line.strip("\n").split(" ")
			list_sims1.append([int(line[0]),[]])
			for i in range(1,len(line)):
				try:
					list_sims1[-1][-1].append(float(line[i]))
				except ValueError:
					print("could not convert to float",line[i],len(line[i]),i,len(line),"with id",line[0])

	with open("Data/Processed/list_sims2.dat",'r') as sims2:
		for line in sims2:
			line = line.strip("\n").split(" ")
			list_sims2.append([int(line[0]),[]])
			for i in range(1,len(line)):
				list_sims2[-1][-1].append(float(line[i]))
	return list_sims1,list_sims2


"""# get the graph features from training, validation or testing set
def get_graph_features(set_name):
    with open("Data/Features/graph_features_"+set_name+".txt", "r") as f:
        reader = csv.reader(f)
        features  = list(reader)
    return features
graph_features_testing = get_graph_features("testing")"""