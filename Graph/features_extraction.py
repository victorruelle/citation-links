#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vizualisation of raw data
"""

import random
import numpy as np
import networkx as nx
import plotly.plotly as py
import plotly.graph_objs as go
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from networkit import *
import math


# data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

training_set_path = '../Data/training_set.txt'
nodes_information_path = '../Data/node_information.csv'
export_training_set = "Data/Processed/edges.csv"
export_nodes_information = "Data/Processed/nodes.csv"



### 1. Reads data and transform it to make it usable by networkit

print("Reading information from nodes...")
# reads articles data
colnames=['id', 'year', 'title', 'authors', 'journal', 'abstract']
nodes = pd.read_csv(nodes_information_path, names=colnames, sep=",", dtype={
                "id" : int,
                "year" : int,
                "title" : str,
                "authors" : str,
                "journal" : str,
                "abstract" :str
            })

# to create the graph later we need to have no breaks in ID values
mapping = {}
reverse_mapping = {}
for i,row in enumerate(nodes.values) :
    mapping[row[0]] = i
    reverse_mapping[i] = row[0]
    row[0] = i

print("Reading data from edges...")
# reads training set
colnames=['source', 'target', 'value']
edges = pd.read_csv(training_set_path, names=colnames, sep = " ", dtype={
                "source" : int,
                "target" : int,
                "value" : int
            })
# changes the IDs of the nodes with the new computed values
for i,row in enumerate(edges.values):
    row[0] = mapping[row[0]]
    row[1] = mapping[row[1]]
    
    
### 2. Splitting the edges into training and validation sets
n_validation = 1000
            

### 3. Create the graph with networkit with just the training_sets

G = Graph()

for i,row in enumerate(nodes.values) :
    G.addNode()
    
training_edges = []
validation_edges = []
    
random_indices = sample([i for i in range(len(mapping)), 50)

for i,row in enumerate(edges.values):
    if row[2] == 1 and i >= n_validation:
        G.addEdge(row[0],row[1])
    if i < n_validation:
        validation_edges.append(row)
    else:
        training_edges.append(row)
 

### 4. Creates the structures for edge transformation to vectors

ADAMIC_ADAR = 0
ADJUSTED_RAND = 1
ALGEBRAIC_DISTANCE = 2
COMMON_NEIGHBORS = 3
JACCARD = 4
KATZ = 5
NEIGHBORHOOD_DISTANCE = 6
NEIGHBORS_MEASURE = 7
PREFERENTIAL_ATTACHMENT = 8
RESOURCE_ALLOCATION = 9
SAME_COMMUNITY = 10
TOTAL_NEIGHBORS = 11
U_DEGREE = 12
V_DEGREE = 13
methods = [ADAMIC_ADAR, ADJUSTED_RAND, ALGEBRAIC_DISTANCE, 
           COMMON_NEIGHBORS, JACCARD, KATZ,
           NEIGHBORHOOD_DISTANCE, NEIGHBORS_MEASURE, PREFERENTIAL_ATTACHMENT,
           RESOURCE_ALLOCATION, SAME_COMMUNITY, TOTAL_NEIGHBORS,
           U_DEGREE, V_DEGREE]
methods=methods[:10]

# class that use a specific method to give a score to an edge in a given graph

class linkPredictor:
   
    def __init__(self,graph,method):
        if method == ADAMIC_ADAR:
            self.predictor = linkprediction.AdamicAdarIndex(graph)
        elif method == ADJUSTED_RAND:
            self.predictor = linkprediction.AdjustedRandIndex(graph)
        elif method == ALGEBRAIC_DISTANCE:
            self.predictor = linkprediction.AlgebraicDistanceIndex(graph,5,5)
            # parameters need to be improved
            self.predictor.preprocess()
        elif method == COMMON_NEIGHBORS:
            self.predictor = linkprediction.CommonNeighborsIndex(graph)
        elif method == JACCARD:
            self.predictor = linkprediction.JaccardIndex(graph)
        elif method == KATZ:
            self.predictor = linkprediction.KatzIndex(graph)
        elif method == NEIGHBORHOOD_DISTANCE:
            self.predictor = linkprediction.NeighborhoodDistanceIndex(graph)
        elif method == NEIGHBORS_MEASURE:
            self.predictor = linkprediction.NeighborsMeasureIndex(graph)
        elif method == PREFERENTIAL_ATTACHMENT:
            self.predictor = linkprediction.PreferentialAttachmentIndex(graph)
        elif method == RESOURCE_ALLOCATION:
            self.predictor = linkprediction.ResourceAllocationIndex(graph)
        elif method == SAME_COMMUNITY:
            self.predictor = linkprediction.SameCommunityIndex(graph)
        elif method == TOTAL_NEIGHBORS:
            self.predictor = linkprediction.TotalNeighborsIndex(graph)
        elif method == U_DEGREE:
            self.predictor = linkprediction.UDegreeIndex(graph)
        elif method == V_DEGREE:
            self.predictor = linkprediction.VDegreeIndex(graph)
            
    def score(self,u,v):
        return self.predictor.run(u,v)
    

# creates an array with all the different predictors
def createAllPredictors(G):
    predictors = []
    for method in methods:
        predictors.append(linkPredictor(G,method))
    return predictors

# return the value of the edge for each of the different scores
def edgeToVector(predictors,u,v):
    result = []
    for predictor in predictors:
        result.append(predictor.score(u,v))
    return result

# return the scores of several edges
maxi = 10000000
def edgesToScore(predictors,edges):
    scores = []
    for i,edge in enumerate(edges):
        if (i%100 == 0):
            print(str(i)+"/"+str(len(edges)))
        s = edgeToVector(predictors,edge[0],edge[1])
        for e in s:
            if math.isnan(e):
                e = maxi
        scores.append(s)
    return scores

predictors = createAllPredictors(G)

### 5. For training set, computes the score values of every edges (even the 0 ones)
    
# limiting the number of testing edges used for training
size_training = 200
print("Converting training edges to vectors...")
training_scores = edgesToScore(predictors,training_edges[:size_training])
print("Training vectors:")
print(training_scores[:10])
# 1 if the edge exists
training_is_edge = [e[2] for e in training_edges[:size_training]]
print("Does the edge exists:")
print(training_is_edge[:10])


### 6. For the validation set, computes the score values of every edges
# limiting the number of edges that will be transformed to vectors
size_validation = 100
print("Converting validation edges to vectors...")
validation_scores = edgesToScore(predictors,validation_edges[:size_validation])
print("Validation vectors:")
print(validation_scores[:10])
validation_real_is_edge = [e[2] for e in validation_edges[:size_validation]]
print("Does the edge exists:")
print(validation_real_is_edge[:10])


### 7. Classification of the validation edges and comparison with the real result

from sklearn.neighbors import KNeighborsClassifier
c = KNeighborsClassifier(3)
c.fit(training_scores,training_is_edge)
score = c.score(validation_scores,validation_real_is_edge)


print('Test accuracy:', score)



### 8. Prediction on the testing set

"""
testing_set_path = "../Data/testing_set.txt"

# reads testing set
colnames=['source', 'target', 'value']
test_edges = pd.read_csv(testing_set_path, names=colnames, sep = " ", dtype={
                "source" : int,
                "target" : int
            })

predictors = createAllPredictors(G)
# half an hour to compute
scores = edgesToScore(predictors,test_edges.values)
"""



    


