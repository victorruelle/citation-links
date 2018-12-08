#!/usr/bin/env python3
# -*- coding: utf-8 -*-





features_training_path = "Data/Features/graph_features_training.csv"
features_validation_path = "Data/Features/graph_features_validation.csv"
nodes_information_path = "Data/node_information.csv"
training_set_path = "Data/Processed/edges_training.txt"
validation_set_path = "Data/Processed/edges_validation.txt"


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
    
# reading features
training_features = pd.read_csv(features_training_path,header=None)
validation_features = pd.read_csv(features_validation_path,header=None)


### 3. Classificiation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


for i,c in enumerate(classifiers):
    print(names[i])
    c.fit(training_features[:20000],[training_edges.values[i][2] for i in range(20000)])
    s = c.score(training_features[:2000],[validation_edges.values[i][2] for i in range(2000)])
    print(s)
