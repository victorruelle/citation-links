#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### Extracts features from edges inside training and validation data


### 1. Definition of the important paramaters of the script

# path to training and validation data
nodes_information_path = "../Data/node_information.csv"
training_set_path = "../Data/Processed/edges_training.txt"
validation_set_path = "../Data/Processed/edges_validation.txt"
testing_set_path = "../Data/testing_set.txt"
features_training = "../Data/Features/graph_features_training.csv"
features_validation = "../Data/Features/graph_features_validation.csv"
features_testing = "../Data/Features/graph_features_testing.csv"


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
print("Reading data from training edges...")
# reads training set
colnames=['source', 'target', 'value']
training_edges = pd.read_csv(training_set_path, names=colnames, sep = " ", dtype={
                "source" : int,
                "target" : int,
                "value" : int
            })   
print("Reading data from validation edges...")
validation_edges = pd.read_csv(validation_set_path, names=colnames, sep = " ", dtype={
                "source" : int,
                "target" : int,
                "value" : int
            })
print("Reading data from testing edges...")
testing_edges = pd.read_csv(testing_set_path, names=colnames[:2], sep = " ", dtype={
                "source" : int,
                "target" : int
            })
    
# changes the IDs of the nodes with the new IDs
def changingIDs(array):
    for i, row in enumerate(array):
        row[0] = mapping[row[0]]
        row[1] = mapping[row[1]]
changingIDs(training_edges.values)
training_edges = training_edges.values
changingIDs(validation_edges.values)
validation_edges = validation_edges.values
changingIDs(testing_edges.values)
testing_edges = testing_edges.values
    
    
    
### 3. Create the graph with the training edges

from networkit import *

# graph on training set
G = Graph()
# graph on testing set
G_ = Graph()

# adding all the nodes
for i,row in enumerate(nodes.values) :
    G.addNode()
    G_.addNode()
    
# adding all the edges (needs to be a 1 as third coordinate)
for i,row in enumerate(training_edges):
    if row[2] == 1:
        G.addEdge(row[0],row[1])
        G_.addEdge(row[0],row[1])
for i,row in enumerate(validation_edges):
    if row[2] == 1:
        G_.addEdge(row[0],row[1])   


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
BETWEENNESS_CENTRALITY = 14
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

# betweenness feature 
betweenness_instance = centrality.Betweenness(G)
betweennes_scores = instance.edgeScores(G)   
betweenness_instance_ = centrality.Betweenness(G_)
betweennes_scores_ = instance.edgeScores(G_)





# return the value of the edge for each of the different scores
maxi = 100000
def edgeToVector(predictors,u,v):
    result = []
    for predictor in predictors:
        s = predictor.score(u,v)
        if math.isnan(s):
            s = maxi
        result.append(s)
    # append other features
    if setting in ['training', 'validation']:
        result.append(betweenness_scores[uv])
    else :
        result.append(betweenness_scores_[uv])
    result.append( 1 if G.hasEdge(u,u) else 0)
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
predictors_ = createAllPredictors(G_)


### 5. Creates function for exporting scores

import csv
def exportScores(name,predictors,edges,output_file):
    print("Converting " + name + " edges to vectors...")
    scores = edgesToScore(predictors,edges)
    
    # saving the result
    with open(output_file, 'w') as myfile:
        wr = csv.writer(myfile, quoting=0)
        for row in scores:
            wr.writerow(row)
    print("Exported " + name + " edges features in " + output_file)

### 6. For training/validation/testing set, computes the score values of every edges (even the 0 ones)

setting = "testing"
if setting == "training":
    exportScores("training",predictors,training_edges,features_training) 
if setting == "validation":
    exportScores("validation",predictors,validation_edges,features_validation) 
if setting == "testing":
    exportScores("testing",predictors_,testing_edges,features_testing)