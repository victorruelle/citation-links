import csv
from collections import defaultdict
from gensim import corpora, models, similarities
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
import numpy
#activate log 
import logging
from sklearn import svm
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


'''
THIS FILE GROUPS ALL THE PREDICTION METHODS
THESE METHODS ARE CALLED BY MAIN.PY

1) SVM CLASSIFIER : a train and a predict method
2) LOGITS CLASSIFIER : a train-predict all-in-one method
3) NN classifier : 
'''

# 1) SVM CLASSIFIER 

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

# 2) LOGITS CLASSIFIER

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
        

# 3) NN Classifier

def create_NN(X_train, Y_train, X_test, Y_test):
	# fix random seed for reproducibility
	numpy.random.seed(7)

	# create model
	model = Sequential()
	#change numbers and number of features
	#Dense is fully connected, maybe make it half connected?
	#change number of layers
	model.add(Dense(12, input_dim=0, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	# Compile model
	#use adaboost as an optimizer? need further investigation
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Fit the model
	model.fit(X_train, Y_train, epochs=150, batch_size=10)

	# evaluate the model
	#scores = model.evaluate(X, Y)
	#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	# calculate predictions
	predictions = model.predict(X_test)
	# round predictions
	rounded = [round(x[0]) for x in predictions]

	#calcule accuracy 
	accuracy = 0
	for i in range(len(predictions)):
		if(predictions[i]==Y_test[i]):
			accuracy+=1
	accuracy/=len(predictions)
	print("Accuracy : ",accuracy)