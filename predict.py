import csv
from collections import defaultdict
from gensim import corpora, models, similarities
from sklearn.linear_model import LogisticRegression
#activate log 
import logging
from sklearn import svm
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_svm(X,Y):
    for x in X:
        for el in x:
            assert type(el)!=str, "found a string in X :"+str(el)+" in "+str(x)
    for y in Y:
        assert type(y)!=str, "found a string in Y : "+str(y)
    clf = svm.SVC(gamma=0.001)
    clf.fit(X,Y)
    return clf

def predict_svm(clf,Y):
    return clf.predict(Y)

def train_predict_logits(testing_set,training_set,list_sims1,list_sims2):
    #initialisation de la liste de listes contenant l'info booléenne: les documents sont-ils liés ou non?
    Y = [[0]*len(list_sims1[0])]*len(list_sims1[0])
    for j in range(len(training_set)):
    #remplir de façon symétrique
        Y[int(training_set[j][1])][int(training_set[j][2])] = int(training_set[j][3])
        Y[int(training_set[j][2])][int(training_set[j][1])] = int(training_set[j][3])
    #maintenant, Y[i][j] == 1 si les docs i et j sont liés, 0 sinon

    #models_XXX[i] contiendra le modèle de régression logistique adaptée au document i pour la caractéristique XXX (titles or abstract)
    models_abstract = []
    models_titles = []	
    for i in range(len(list_sims1[0])):
        Xi1 = list_sims1[i][1]
        Xi2 = list_sims2[i][1]
        Yi = Y[i]
        
        models_abstract.append(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(Xi1, Yi))
        models_titles.append(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(Xi2, Yi))

    #on process le testing set pour qu'il soit sous forme de similarités
    test_sims_abstract = []
    test_sims_titles = []

    for j in range(len(testing_set)):
        test_sims_abstract.append(list_sims1[int(testing_set[j][1])][int(testing_set[j][2])])
        test_sims_titles.append(list_sims2[int(testing_set[j][1])][int(testing_set[j][2])])	

    #predictions_abstract[i] et predictions_titles[i] vont contenir les predicitions pour le testing set en fonction du modèle adapté au document i 
    predictions_abstract = []
    predictions_titles = []
    for i in range(len(list_sims1[0])):
        predictions_abstract.append(models_abstract[i].predict(test_sims_abstract))
        predictions_titles.append(models_titles[i].predict(test_sims_titles))
        