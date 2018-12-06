#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### Split the data into training and validation set
### Saves the result in Data/Processed folder

### !!! BE CAREFUL, computing this script takes a lot of time

import pandas as pd


### 1. Definition of important variables

# Parameters for splitting
percentage_validation = 0.05 # percentage of edges used for validation

# Definition of some path variables
training_set_path = '../Data/training_set.txt'
export_training_set = "../Data/Processed/edges_training.txt"
export_validation_set = "../Data/Processed/edges_validation.txt"


### 2. Reads edges data

print("Reading data from edges...")
colnames=['source', 'target', 'value']
edges = pd.read_csv(training_set_path, names=colnames, sep = " ", dtype={
                "source" : int,
                "target" : int,
                "value" : int
            })


### 3. Splitting

n = len(edges.values)
n_validation = int(percentage_validation * n)

# takes n_validation random edges
import random
print("Choosing randomly %d edges for validation" % n_validation)
indices = random.sample([i for i in range(n)],n_validation)
training_edges, validation_edges = [], []
for i, row in enumerate(edges.values):
    if i % 10000 == 0:
        print(str(i)+"/"+str(n))
    if i in indices:
        validation_edges.append(row)
    else:
        training_edges.append(row)


### 4. Exporting the result 

import csv

# exporting training set
with open(export_training_set, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=" ", quoting=0)
    for row in training_edges:
        wr.writerow([str(row[0]),str(row[1]),str(row[2])])
print("Saved training edges in " + export_training_set)

# exporting validation set
with open(export_validation_set, 'w') as myfile:
    wr = csv.writer(myfile, delimiter=" ", quoting=0)
    for row in validation_edges:
        wr.writerow([str(row[0]),str(row[1]),str(row[2])])
print("Saved validation edges in " + export_validation_set)