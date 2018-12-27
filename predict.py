import csv
from collections import defaultdict
from gensim import corpora, models, similarities
import sklearn.utils.testing as t
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

# 0) DEFINITION OF A GENERIC CLASSIFIER CLASS

class Classifier():

    def __init(self):
        # define random seed to get the same results every time
        numpy.random.seed(7)
        self.fitted = False

    def fit(self,X_training,y_training):
        return
    # will be defined in child classes

    def predict(self,X):
        return
    # will be defined in child classes

    def accuracy(self,X,y):
        predictions = self.predict(X)
        accuracy = 0
        for i in range(len(predictions)):
            if (predictions[i] == y[i]):
                accuracy += 1
        accuracy /= len(predictions)
        print("Accuracy : ", accuracy)
        return accuracy


# 1) SVM CLASSIFIER

class SVMClassifier(Classifier):
    def __init__(self,gamma=0.001,features="all"):
        # features is here to restrain the features used by the classifier
        self.features = "all"
        self.fitted = False
        self.n_features = -1
        self.classifier = svm.SVC(gamma=gamma)

    def __features_to_array(self,start):
        # start is an array with the ids of the features ([0,...,n_features-1])
        # transforms it to keep only the feature defines in self.features
        if self.features == "all":
            return start
        else:
            # we assume there the features is an array
            return self.features

    def fit(self,X_training,y_training):
        self.fitted = True
        selected_features = self.__features_to_array(list(range(X_training.shape[0])))
        self.classifier.fit(X_training[selected_features],y_training)

    def predict(self,X):
        if not self.fitted:
            print("ERROR: model not fitted")
            return
        return self.classifier.predict()


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

class NNClassfier:
    def __init__(self,n_input,size_layers):# fix random seed for reproducibility

        # creates a network with len(size_layers) + 2 layers

        # create model
        self.model = Sequential()
        #Dense is fully connected, maybe make it half connected?
        self.model.add(Dense(size_layers[0], input_dim=n_input, activation='relu'))
        for i in range(1,len(size_layers)):
            self.model.add(Dense(size_layers[i], activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile model
        #use adaboost as an optimizer? need further investigation
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.fitted = False

    def fit(self,X_train,y_train,epochs=150):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=10)
        self.fitted = True

    def predict(self,X):
        if not self.fitted:
            print("ERROR: model not fitted")
            return
        predictions = self.model.predict(X)
        rounded = [int(round(x[0])) for x in predictions]
        return rounded


