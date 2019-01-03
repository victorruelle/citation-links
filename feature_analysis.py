from info_gain.info_gain import info_gain as ig
import numpy as np
import pandas as pd

def rank_features(X,Y,features_info):
    # returns a list assigning each feature with its information gain 
    X = np.array(X)
    ranks = []
    for i in range(X.shape[1]):
        ranks.append([features_info[i][1],ig(X[:,i],Y)])
    ranks.sort(key = lambda rank : rank[1], reverse = True)
    return ranks

def correlations(features,features_info):
    features = np.array(features)
    cols = [ f[1] for f in features_info ]
    df = pd.DataFrame(data = features, columns = cols, dtype = np.float64)
    corr_matrix = df.corr().abs()
    correlated = []
    ids = list(corr_matrix.keys())
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            feature1,feature2 = ids[i],ids[j]
            if corr_matrix[feature1][feature2] > 0.75 : 
                correlated.append([feature1+" x "+feature2,corr_matrix[feature1][feature2]])
    correlated.sort(key = lambda c : c[1], reverse = True)
    return correlated

