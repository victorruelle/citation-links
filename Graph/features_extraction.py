#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### Extracts features from edges inside training and validation data


### 1. Definition of the important paramaters of the script

# path to training and validation data
nodes_information_path = "../Data/node_information.csv"
training_set_path = "../Data/Processed/edges_training.txt"
validation_set_path = "../Data/Processed/edges_validation.txt"
features_training = "../Data/Features/graph_features_training.csv"
features_validation = "../Data/Features/graph_features_validation.csv"


### 2. Reads edge and node data and insert it in pandas DataFrame

import pandas as pd

# Reads information from nodes
print("Reading information from nodes...")
colnames=['id', 'year', 'title', 'authors', 'journal', 'abstract']
nodes = pd.read_csv(nodes_information_path, names=colnames, sep=",", dtype={
                "id" : int,
                "year" : int,
                "title" : str,
                "authors" : str,
                "journal" : str,
                "abstract" :str
            })

# creates a mapping from the nodes id to [0, number of nodes - 1]
mapping = {}
reverse_mapping = {}
for i,row in enumerate(nodes.values) :
    mapping[row[0]] = i
    reverse_mapping[i] = row[0]
    row[0] = i

# reading data from edges
print("Reading data from edges...")
# reads training set
colnames=['source', 'target', 'value']
training_edges = pd.read_csv(training_set_path, names=colnames, sep = " ", dtype={
                "source" : int,
                "target" : int,
                "value" : int
            })
validation_edges = pd.read_csv(validation_set_path, names=colnames, sep = " ", dtype={
                "source" : int,
                "target" : int,
                "value" : int
            })
    
# changes the IDs of the nodes with the new IDs
for i,row in enumerate(training_edges.values):
    row[0] = mapping[row[0]]
    row[1] = mapping[row[1]]
training_edges = training_edges.values
for i,row in enumerate(validation_edges.values):
    row[0] = mapping[row[0]]
    row[1] = mapping[row[1]]
validation_edges = validation_edges.values
    
    
    
### 3. Create the graph with the training edges

from networkit import *

G = Graph()

# adding all the nodes
for i,row in enumerate(nodes.values) :
    G.addNode()
    
# adding all the edges (needs to be a 1 as third coordinate)
for i,row in enumerate(training_edges):
    if row[2] == 1:
        G.addEdge(row[0],row[1])


### 4. Creates the structures for edge transformation to vectors
        
import math

# List of all methods that can be used to give a score to an edge
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
# many methods
methods = [ADAMIC_ADAR, ADJUSTED_RAND, ALGEBRAIC_DISTANCE, 
           COMMON_NEIGHBORS, JACCARD, KATZ,
           NEIGHBORHOOD_DISTANCE, NEIGHBORS_MEASURE, PREFERENTIAL_ATTACHMENT,
           RESOURCE_ALLOCATION, SAME_COMMUNITY, TOTAL_NEIGHBORS,
           U_DEGREE, V_DEGREE]
# less methods (jaccard and katz take a lot of time)
methods = [ADAMIC_ADAR, ADJUSTED_RAND,COMMON_NEIGHBORS,
           NEIGHBORHOOD_DISTANCE, NEIGHBORS_MEASURE, PREFERENTIAL_ATTACHMENT,
           RESOURCE_ALLOCATION, SAME_COMMUNITY, TOTAL_NEIGHBORS,
           U_DEGREE, V_DEGREE]

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
maxi = 100000
def edgeToVector(predictors,u,v):
    result = []
    for predictor in predictors:
        s = predictor.score(u,v)
        if math.isnan(s):
            s = maxi
        result.append(s)
    return result

# return the scores of several edges
def edgesToScore(predictors,edges):
    scores = []
    for i,edge in enumerate(edges):
        if (i%10000 == 0):
            print(str(i)+"/"+str(len(edges)))
        s = edgeToVector(predictors,edge[0],edge[1])
        scores.append(s)
    return scores

predictors = createAllPredictors(G)


### 5. For training set, computes the score values of every edges (even the 0 ones)
    
import csv
size_viz = 10
# limiting the number of testing edges used for training
print("Converting training edges to vectors...")
training_scores = edgesToScore(predictors,training_edges)
print("Training vectors:")
print(training_scores[:size_viz])

# saving the result
with open(features_training, 'w') as myfile:
    wr = csv.writer(myfile, quoting=0)
    for row in training_scores:
        wr.writerow(row)
print("Exported training edges features in " + features_training)


### 6. For the validation set, computes the score values of every edges

# limiting the number of edges that will be transformed to vectors
print("Converting validation edges to vectors...")
validation_scores = edgesToScore(predictors,validation_edges)
print("Validation vectors:")
print(validation_scores[:size_viz])

# saving the result
with open(features_validation, 'w') as myfile:
    wr = csv.writer(myfile, quoting=0)
    for row in validation_scores:
        wr.writerow(row)
print("Exported training edges features in " + features_validation)


    


