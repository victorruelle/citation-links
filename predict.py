import csv
from collections import defaultdict
from gensim import corpora, models, similarities
import sklearn.utils.testing as t
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
import numpy
import logging
from sklearn import svm
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np

#activate log 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


'''
THIS FILE GROUPS ALL THE PREDICTION METHODS
THESE METHODS ARE CALLED BY MAIN.PY

0) DEFINITION OF A CLASSIFIER : creates a template class for all the prediction models
1) SVM CLASSIFIER : a svm classifier
2) LOGITS CLASSIFIER : a train-predict all-in-one method
<<<<<<< HEAD
3) NN classifier : neural network
=======
3) NN CLASSIFIER : a neural network method
>>>>>>> 63e684fa18689b8759c4ab511449c033d9b140d5
'''

# 0) DEFINITION OF A GENERIC CLASSIFIER CLASS

class Classifier:

	def __init__(self,features):
		# define random seed to get the same results every time
		numpy.random.seed(7)
		self.fitted = False
		self.features = features

	def features_to_array(self,start):
		# start is an array with the ids of the features ([0,...,n_features-1])
		# transforms it to keep only the feature defines in self.features
		if self.features == "all":
			return start
		else:
			# we assume there the features is an array
			return self.features

	def fit(self,X_training,y_training):
		return
	# will be defined in child classes

	def predict(self,X):
		return
	# will be defined in child classes

	# test accuracy of the model on (X,y)
	def score(self,X,y):
		predictions = self.predict(X)
		accuracy = 0
		for i in range(len(predictions)):
			if (predictions[i] == y[i]):
				accuracy += 1
		accuracy /= len(predictions)
		print("Accuracy : ", accuracy)
		return accuracy

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

# 1) SVM CLASSIFIER

class SVMClassifier(Classifier):
	def __init__(self,gamma=0.001,features="all"):
		# features is here to restrain the features used by the classifier
		super().__init__(features)
		self.gamma = gamma
		self.classifier = svm.SVC(gamma=gamma)

	def fit(self,X_training,y_training):
		self.fitted = True
		self.selected_features = self.features_to_array(list(range(X_training.shape[1])))
		self.classifier.fit(X_training[:,self.selected_features],y_training)

	def predict(self,X):
		if not self.fitted:
			print("ERROR: model not fitted")
			return
		return self.classifier.predict(X[:,self.selected_features])

	def get_params(self, deep=True):
		return {"gamma": self.gamma}



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

class NNClassifier(Classifier):
	def __init__(self,n_input,size_layers,features="all"):
		super().__init__(features)

		self.n_input = n_input
		self.size_layers = size_layers

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


	def fit(self,X_train,y_train,epochs=2):
		self.fitted = True
		self.selected_features = self.features_to_array(list(range(X_train.shape[1])))
		self.model.fit(X_train[:,self.selected_features], y_train, epochs=epochs, batch_size=10)

	def predict(self,X):
		if not self.fitted:
			print("ERROR: model not fitted")
			return
		predictions = self.model.predict(X[:,self.selected_features])
		rounded = [int(round(x[0])) for x in predictions]
		return rounded

	def get_params(self, deep=True):
		return {"n_input": self.n_input, "size_layers": self.size_layers}



class xgb_classifier(Classifier):
	def __init__(self,
				 parameters={},
				 learning_rate=0.01,
				 colsample_bytree=1,
				 subsample=0.8,
				 n_estimators=800,
				 max_depth=3,
				 gamma=1,features="all"):
		super().__init__(features)
		# fix random seed for reproducibility
		numpy.random.seed(7)
		self.parameters = parameters
		self.model = xgb.XGBClassifier(#silent=parameters["silent"],
						scale_pos_weight=parameters["scale_pos_weight"],
						learning_rate=learning_rate,
						colsample_bytree = colsample_bytree,
						subsample = subsample,
						objective= parameters["objective"], 
						n_estimators= n_estimators,
						reg_alpha = parameters["reg_alpha"],
						max_depth= max_depth,
						gamma= gamma)
		self.fitted = False

	def fit(self,X_train,y_train):
		data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
		params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.005,'max_depth': 5, 'alpha': 10, 'nthread':4}
		cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5, num_boost_round=50,early_stopping_rounds=10,metrics="auc", as_pandas=True, seed=123)
		print(cv_results.head())
		eval_metric = ["auc","error"]
		self.selected_features = self.features_to_array(list(range(X_train.shape[1])))
		self.model.fit(X_train[:,self.selected_features],y_train,eval_metric=eval_metric)
		self.fitted = True
		
	def test(self,X_train,y_train,X_test,y_test):
		eval_set = [(X_train[:,self.selected_features], y_train), (X_test[:,self.selected_features], y_test)]
		eval_metric = ["auc","error"]
		self.model.fit(X_train[:,self.selected_features], y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)


	def predict(self,X):
		if not self.fitted:
			print("ERROR: model not fitted")
			return
		predictions = self.model.predict(X[:,self.selected_features])
		return predictions

	def evaluate(self,preds,y_test):
		return np.sqrt(mean_squared_error(y_test, preds))

	def get_params(self, deep=True):
		return self.parameters
