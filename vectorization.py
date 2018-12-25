import csv
from collections import defaultdict
from gensim import corpora, models, similarities
from sklearn.linear_model import LogisticRegression
#activate log 
import logging
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


'''

TL;DR
get_features_of_set(name,metas) to get the features vector for a given set
 - name: training|validation|testing
 - metas: IDs, years, authors, corpus_abstract, corpus_title

THIS FILE CONTAINS METHODS TO COMPUTE DESCRIPTORS FOR OUR DATA

THE MAIN METHODS ARE
	1) get_features_of_set(name,metas) : takes a data_set (training/validation/testing) and computes the entire X feature vector
  		- it is done to save as much computed data as possible
  			for a given set, meta features, text features and graph features are saved in separate files
  			they are then gathered to form a features file (for each dataset)
  			if the meta/text/graph file doesn't exist, they are computed
  		- if the function can't find the data saved, it will generate it and save it

OTHER 'SUPPORT' METHODS:
	1) comput_similarities : Jo's code, computes sims between titles and abstracts (long)
	2) save_sims : save similarity matrices from Jo's code. DO NOT USE as it as (10Go files...).
		WILL ERASE PREVIOUS SAVES WHEN CALLED!
	3) recover_list_sims : recover saved sims 
'''


def get_features_of_set(name,metas):
	# if the file containing all the features for the data set exsits, load the data
	# otherwise, call gather_features_of_set that will create the file with the wanted data

	if not os.path.isfile("Data/Features/all_features_%s.csv"%name):
		print("The file corresponding to %s was not existing, creates it."%name)
		gather_features_of_set(name,metas)

	file = open("Data/Features/all_features_%s.csv"%name,"r")
	reader = csv.reader(file)
	features = list(reader)
	features = [[float(e) for e in f] for f in features]

	return features

def gather_features_of_set(name,metas):
	# saves all the features of a given set (training|testing|validation) in a csv file
	# gets the features if they already have been created ; creates them otherwise
	print("Beginning of %s_set treatment" %name)

	if os.path.isfile("Data/Features/all_features_%s.csv"%name):
		print("Features were already gathered for %s. Please delete Data/Features/all_features_%s.csv to remake the gathering phase")
		return

	# gets the features computed from the graph study
	graph_features_path = "Data/Features/graph_features_%s.csv"%name
	if os.path.isfile(graph_features_path):
		# if features data has already been computed and saved, gets it
		file = open(graph_features_path,"r")
		reader = csv.reader(file)
		graph_features = list(reader)
		graph_features = [[float(sub_element) for sub_element in element] for element in graph_features]
		print("Graph features have been loaded for %s"%name)
	else:
		# if it has not, prints an error message
		print("Graph features have not been saved in %s... They will be generated now..." % graph_features_path)
		graph_features = generate_graph_features(name,metas)

	# gets the features computed from texts
	text_features_path = "Data/Features/text_features_%s.csv"%name
	if os.path.isfile(text_features_path):
		# if features data has already been computed and saved, gets it
		file = open(text_features_path, "r")
		reader = csv.reader(file)
		text_features = list(reader)
		text_features = [[float(sub_element) for sub_element in element] for element in text_features]
		print("Text features have been loaded for %s" % name)
	else:
		# if it has not, prints an error message
		print("Text features have not been saved in %s... They will be generated now..."%text_features_path)
		text_features = generate_text_features(name,metas)

	# gets the features computed from meta data
	meta_features_path = "Data/Features/meta_features_%s.csv"%name
	if os.path.isfile(meta_features_path):
		# if features data has already been computed and saved, gets it
		file = open(meta_features_path, "r")
		reader = csv.reader(file)
		meta_features = list(reader)
		meta_features = [[float(sub_element) for sub_element in element] for element in meta_features]
		print("Meta features have been loaded for %s" % name)
	else:
		# if it has not, prints an error message
		print("Meta features have not been saved in %s... They will be generated now..."%meta_features_path)
		meta_features = generate_meta_features(name,metas)

	# gather all the data and saved it
	saving_path = "Data/Features/all_features_%s.csv"%name
	assert len(meta_features) == len(text_features) and len(meta_features) == len(graph_features)
	features = [meta_features[i]+graph_features[i]+text_features[i] for i in range(len(graph_features))]
	file = open(saving_path,'w')
	writer = csv.writer(file)
	for edge in features:
		writer.writerow(edge)
	print("All the features have been gathered and saved in %s" % name)


def loads_edges(type):
	# loads edge data for a given set
    # type : testing|training|validation
	path = "Data/Processed/edges_%s.txt"%type
	with open(path, "r") as f:
		reader = csv.reader(f)
		res = list(reader)
	# each element: edgeA(int) edgeB(int) (value(0/1) if training)
	res = [[int(sub_element) for sub_element in element[0].split(" ")]
                    for element in res]
	# nothing to change with node_info
	return res


def generate_meta_features(name,metas):
	# generate all the features derived from metadata of the set name and saves them in Data/Features/meta_...

	edges = loads_edges(name)

	# metas contain all the information lists
	IDs,years,authors,_,_ = metas

	features = []
	for edges in edges:
		u,v=edges[0],edges[1]
		x = []
		# converting to the indexing used in the feature lists
		id1 = IDs.index(u)
		id2 = IDs.index(v)

		# adding years difference
		x.append(abs(years[id1]-years[id2]))

		# adding number of common authors ( we can learn author habits )
		a1 = authors[id1]
		a2 = authors[id2]
		n = len(set(a1).intersection(set(a2)))
		x.append(n)

		features.append(x)

	saving_path = "Data/Features/meta_features_%s.csv"%name
	file = open(saving_path, 'w')
	writer = csv.writer(file)
	for edge in features:
		writer.writerow(edge)
	print("Saved meta features in %s"%saving_path)

	return features

def generate_text_features(name,metas):
	# get all the lists inside meta
	IDs, years, authors, corpus_abstract, corpus_title = metas
	sims_abstract, sims_title = compute_similarities(corpus_abstract, corpus_title)

	# load edges from the given data set
	edges = loads_edges(name)

	features = []

	for edge in edges:
		u,v = edge[0],edge[1]
		x = []

		# converting to the indexing used in the feature lists
		id1 = IDs.index(u)
		id2 = IDs.index(v)

		# adding the similarities between titles and descriptions
		x.append(sims_abstract[id1][1][id2])
		x.append(sims_title[id1][1][id2])

		features.append(x)

	saving_path = "Data/Features/text_features_%s.csv" % name
	file = open(saving_path, 'w')
	writer = csv.writer(file)
	for edge in features:
		writer.writerow(edge)
	print("Saved text features in %s"%saving_path)

	return features

def generate_graph_features(name,metas):
	# TODO import code from Graph/features/extraction.py
	print("ERROR: not supported yet")


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



