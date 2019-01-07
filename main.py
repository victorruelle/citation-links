import vectorization as vect
import predict as pred
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from feature_analysis import rank_features

'''
MAIN FILE

1. CONTAINS : 
    1.1) CODE TO READ THE DATA AND PLACE IT IN USABLE LISTS
    1.1) METHODS TO GET PREDICTIONS USING VECTORIZATION METHODS FROM vectorization.py
    AND PREDICTION METHODS FROM prediction.py
    1.2) METHOD TO SAVE A PREDICTION IN THE RIGHT FORMAT
    
2. DATA MINING AND VISUALIZATION
    2.1) Plot a feature vs. another one with the chosen format

3. WHEN main.py IS CALLED :
    1) READS THE DATA AND PLACES IT IN USABLE LISTS
    2) EXECUTES WHATEVER IS IN THE if __name__=='__main__' LOOP
'''


#  1.1) READING DATA AND PLACING IT IN USABLE LISTS

def loads_data(path,type):
    # path is the path of the file to read
    # type : testing|training|node_info
    with open(path, "r") as f:
        reader = csv.reader(f)
        res = list(reader)
    if type == "training" or type == "testing":
        # each element: edgeA(int) edgeB(int) (value(0/1) if training)
        res = [ [int(sub_element) for sub_element in element[0].split(" ")]
                    for element in res]
    # nothing to change with node_info
    return res

#here we open and process the testing set, ie pairs of articles without a link
print("Loading testing set...")
testing_set = loads_data("Data/Processed/edges_testing.txt","testing")
#we obtain a list of lists of 2 strings

#here we pre-process the training and validation data
print("Loading training set...")
training_set = loads_data("Data/Processed/edges_training.txt","training")
print("Loading validation set...")
validation_set = loads_data("Data/Processed/edges_validation.txt","training")
#we obtain a list of lists of 3 strings: the two articles IDs and 1 if there is a link, 0 otherwise

print("Loading node information...")
node_info = loads_data("Data/node_information.csv","node_info")
# create a list with list with all the words for each info
IDs = [int(element[0]) for element in node_info]
years = [int(element[1]) for element in node_info]
corpus_title = [element[2].split(" ") for element in node_info]
authors = [element[3].split(" ") for element in node_info]
journals = [element[4] for element in node_info]
corpus_abstract = [element[5].split(" ") for element in node_info]
metas = [IDs,years,authors,corpus_abstract,corpus_title,journals]



############################
# 1.2) METHODS TO GET PREDICTIONS USING VECTORIZATION METHODS FROM vectorization.py
#      AND PREDICTION METHODS FROM prediction.py


# SVM prediction
def get_predictions_svm():
    print("---- running...")
    list_sims1, list_sims2 = vect.compute_similarities(corpus_abstract,corpus_title)
    print("---- similarity matrices computed")
    data_set,Y_train = [],[]
    for id1,id2,y in training_set:
        Y.append(y)
        data_set.append([id1,id2])
    X_train = vect.features_all(data_set,metas)
    print("---- all features have been computed")
    model = pred.train_svm(X_train,Y_train)
    print("---- svm model has been trained")
    X_test = testing_set
    Y_test = pred.predict_svm(model,X_test)
    print("---- predictions have been made")
    return(Y_test)

def get_predictions_NN():
    print("---- running...")
    list_sims1, list_sims2 = vect.compute_similarities(corpus_abstract,corpus_title)
    print("---- similarity matrices computed")
    data_set,Y_train = [],[]
    for id1,id2,y in training_set:
        Y_train.append(y)
        data_set.append([id1,id2])
    X_train = vect.features_all(data_set,metas)
    print("---- all features have been computed")
    X_test = vect.features_all(testing_set,metas)
    NN_pred = pred.create_NN(X_train,Y_train,X_test)
    return(NN_pred)

############################
# 1.3) METHOD TO SAVE PREDICITONS IN THE RIGHT FORMAT

def save_predictions(Y):
    # saves a prediction list in the right format
    # assumes predictions have been made in the "natural" order (that of testing_set)
    name = "predictions_"+str(datetime.now().day)+"_"+str(datetime.now().month)+"_"+str(datetime.now().hour)+"h"+str(datetime.now().minute)+"m"+str(datetime.now().second)
    with open("Predictions/"+name+".txt",'w') as out:
        out.write("id,category\n")
        for i in range(len(Y)):
            out.write(str(i)+","+str(int(Y[i]))+"\n")
    print("prediction successfully written to Predictions/"+name)
    return name

def log_predictions(file_name,general_params,method_params,accuracy):
    log_file = "Predictions/log_predictions.txt"
    headers = "accuracy;prediction_file;general_params;method_params\n"
    if not os.path.isfile(log_file):
        open(log_file,"w").write(headers)
    # the result is added to the log file to keep a trace of every test
    file = open(log_file,"a")
    file.write("%f;%s;%s;%s\n" % (accuracy,file_name,str(general_params),str(method_params)))

def test(general_params,method_params,X_training,y_training,X_validation,y_validation,X_testing):
    print("entering test function...")
    # test a given set of params and save the output"
    assert "method" in general_params.keys()
    n_training = general_params["n_training"]
    n_validation = general_params["n_validation"]
    print(general_params)
    selected_features = general_params["selected_features"]
    # takes random part of the data
    permutation_training = np.random.permutation(len(X_training))[:n_training]
    permutation_validation = np.random.permutation(len(X_validation))[:n_validation]
    X_training = X_training[permutation_training,:]
    y_training = y_training[permutation_training]
    X_validation = X_validation[permutation_validation,:]
    y_validation = y_validation[permutation_validation]
    print(general_params)
    print(method_params)
    # For each method, create the model with the specific parameters
    if general_params["method"] == "NN":
        size_layers = method_params["size_layers"]
        epochs = method_params["epochs"]
        if selected_features == "all":
            size_input = X_training.shape[1]
        else:
            size_input = len(selected_features)
        model = pred.NNClassifier(size_input,size_layers,features=selected_features)
        model.fit(X_training,y_training,epochs=epochs)
    if general_params["method"] == "SVC":
        gamma = method_params["gamma"]
        model = pred.SVMClassifier(gamma=gamma,features=selected_features)
        model.fit(X_training,y_training)
    if general_params["method"] == "XGB":
        model = pred.xgb_classifier(method_params,features=selected_features)
        model.fit(X_training,y_training)
    # Test the model, saves a prediction for the testing set and log everything
    accuracy = model.score(X_validation,y_validation)
    print(accuracy)
    pred_Y = model.predict(X_testing)
    pred_file = save_predictions(pred_Y)
    log_predictions(pred_file,method_params,general_params,accuracy)
    return accuracy


### 2. DATA MINING AND VIZ

# informations on features
# id, what it is, keep it
features_info = [
    [0,"(Graph) Adamic adar",True],
    [1,"(Graph) Adjusted rand",True],
    [2,"(Graph) Common Neighbors",True],
    [3,"(Graph) Neighborhood distance",True],
    [4,"(Graph) Neighbors measure",True],
    [5,"(Graph) Prefential attachment",True],
    [6,"(Graph) Ressource allocation",True],
    [7,"(Graph) Same community",True],
    [8,"(Graph) Total neighbors",True],
    [9,"(Graph) U degree",True],
    [10,"(Graph) V degree",True],
    [11,"(Graph) Jaccard index",True],
    [12,"(Meta) Delta publication year",True],
    [13,"(Meta) Number of common authors",True],
    [14,"(Text) Abstract similitude",True],
    [15,"(Text) Title similitude",True]
]

def confront_features(features,labels,id1,id2,plot_type,name):
    # plot_type : scatter_plot
    positives, negatives = [], []
    for i,l in enumerate(labels):
        if l == 1:
            positives.append(features[i])
        else:
            negatives.append(features[i])
    positives = np.array(positives)
    negatives = np.array(negatives)
    if plot_type == "scatter_plot":
        plt.scatter(positives[:,id1],positives[:,id2],marker="x",c="green")
        plt.scatter(negatives[:,id1],negatives[:,id2],marker="x",c="red")
        plt.savefig("%s.png"%name)
        plt.show()


from sklearn import preprocessing
def scaling_features(X_ref,X_to_scale):
    std_scale = preprocessing.StandardScaler().fit(X_ref)
    return(std_scale.transform(X_to_scale))

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
def feature_selection(X,y):
    model = LogisticRegression()
    # Adding cross validation to choose the best number of features
    rfe = RFECV(model,verbose=1,n_jobs=4)
    rfe = rfe.fit(X, y)
    features = []
    for i,e in enumerate(rfe.support_):
        if e: features.append(i)
    return(features)


if (__name__ == "__main__"):

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    normalization = False
    n_features = 8

    X_training = np.array(vect.get_features_of_set("training",metas))
    if normalization:
        X_training = scaling_features(X_training,X_training)
    y_training = np.array([e[2] for e in training_set])
    X_validation = np.array(vect.get_features_of_set("validation",metas))
    # scaling based on training set
    if normalization:
        X_validation = scaling_features(X_training,X_validation)
    y_validation = np.array([e[2] for e in validation_set])
    X_testing = np.array(vect.get_features_of_set("testing",metas))
    # scaling based on training set
    if normalization:
        X_validation = scaling_features(X_training,X_validation)

    #features_logistic_regression = feature_selection("logistic_regression",X_training,y_training)
    #features_logistic_regression = [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    #print(features)

    # features sorted according to information gain
    # TODO how it is computed ?
    best_features_IG = [15, 6, 14, 2, 8, 5, 3, 4, 9, 7, 13, 10, 12, 11, 0, 1]
    # removing features that where not taken in logistic regression
    # features_logistic_regression = feature_selection(X_training, y_training)
    features_logistic_regression = [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    features = []
    for i in best_features_IG:
        if i in features_logistic_regression:
            features.append(i)


    # plot accuracy of the different models depending on the number of features used
    def plot_accuracy_features(best_features, name):
        range = range(1, len(best_features))
        n_training = 20000
        n_validation = 3000
        acc_SVC, acc_NN, acc_XGB = [], [], []
        for i in range:
            general_params_SVC = {"method": "SVC", "n_training": n_training, "n_validation": n_validation,
                                      "selected_features": best_features[:i]}
            method_params_SVC = {"gamma": 0.0005}
            acc_SVC.append(
                    test(general_params_SVC, method_params_SVC, X_training, y_training, X_validation, y_validation,
                         X_testing))
            general_params_NN = {"method": "NN", "n_training": n_training, "n_validation": n_validation,
                                     "selected_features": best_features[:i]}
            method_params_NN = {"size_layers": [20, 20, 20], "epochs": 3}
            acc_NN.append(
                    test(general_params_NN, method_params_NN, X_training, y_training, X_validation, y_validation,
                         X_testing))
            general_params_XGB = {"method": "XGB", "n_training": n_training, "n_validation": n_validation,
                                      "selected_features": best_features[:i]}
            method_params_XGB = {"silent": True,
                                     "scale_pos_weight": 1,
                                     "learning_rate": 0.001,
                                     "colsample_bytree": 1,
                                     "subsample": 0.8,
                                     "objective": 'binary:logistic',
                                     "n_estimators": 500,
                                     "reg_alpha": 0.3,
                                     "max_depth": 4,
                                     "gamma": 1}
            acc_XGB.append(
                    test(general_params_XGB, method_params_XGB, X_training, y_training, X_validation, y_validation,
                         X_testing))
        plt.plot(range, acc_NN, color="blue", label="Neural Network")
        plt.plot(range, acc_SVC, color="red", label="SVC")
        plt.plot(range, acc_XGB, color="green", label="XGBoost")
        plt.legend(
                "Accuracy of the different models depending on the number of features used (sorted on decreasing information gain)")
        plt.savefig(name)
        plt.show()


    #plot_accuracy_features(features, "Analysis/Accuracy(nb_features).png")
    # deciding n_features on infomation gain and previous plot
    n_features = 8
    # taking the n_features best features
    features = features[:n_features]
    print("Features used:")
    for i, f in enumerate(features):
        print("\t%d: %s" % (i, features_info[f][1]))
    print()


    features = list(range(16))

    #ranks = rank_features(np.array([y_training,y_training]).transpose(),y_training,[[0,"label",True],[1,"label",True]])
    #print(ranks)
    


    """confront_features(X_training[:2000],y_training[:2000],11,12,"scatter_plot","Delta year vs. Common authors")
    confront_features(X_training[:2000],y_training[:2000],13,14,"scatter_plot","Abstract similitude vs. Title similitude")
    confront_features(X_training[:2000],y_training[:2000],3,4,"scatter_plot","Neighborhood distance vs. Neighbors measure")
    confront_features(X_training[:2000],y_training[:2000],2,8,"scatter_plot","Common neighbors vs. Total Neighborhood")"""


    """ TO PLOT ACCURACY VS GAMMA
    general_params = {"method":"SVC","n_training":20000,"n_validation":3000}
    name = "SVM_comparison"+str(datetime.now().day)+"_"+str(datetime.now().month)+"_"+str(datetime.now().hour)+"h"+str(datetime.now().minute)+"m"+str(datetime.now().second)
    gammas = np.arange(0.0005,0.005,0.001)
    accuracies = []
    for gamma in gammas:
        method_params = { "gamma" : gamma, "selected_features" : "all"}
        accuracies.append(test(general_params,method_params,X_training,y_training,X_validation,y_validation,X_testing))
    plt.scatter(gammas, accuracies)
    plt.savefig("%s.png"%name)
    plt.show()
    """

    """
    # EXAMPLE FOR NN
    general_params = {"method":"NN","n_training":20000,"n_validation":3000,"selected_features":features,"normalization":normalization}

    method_params = { "size_layers" : [20,20,20], "epochs" : 2}
    test(general_params,method_params,X_training,y_training,X_validation,y_validation,X_testing)
    """

    """
    # EXAMPLE FOR SVM
    general_params = {"method":"SVC","n_training":20000,"selected_features":features,"n_validation":3000,"normalization":normalization}
    method_params = { "gamma" : 0.0005, "selected_features" : "all"}
    test(general_params,method_params,X_training,y_training,X_validation,y_validation,X_testing)
    """


    # EXAMPLE FOR XGBOOST
    general_params = {"method":"XGB","n_training":20000,"n_validation":3000,"selected_features" : "all"}
    method_params = {
                     'colsample_bytree': 1,
                     'gamma': 1,
                     'learning_rate': 0.0005,
                     'max_depth': 8,
                     'n_estimators': 1500,
                     'objective': 'binary:logistic',
                     'reg_alpha': 0.3,
                     'scale_pos_weight': 0.9,
                     'silent': True,
                     'subsample': 0.8}
    test(general_params,method_params,X_training,y_training,X_validation,y_validation,X_testing)
	
	
	# fine tuning xgboost
"""
	from sklearn.model_selection import GridSearchCV
	import scipy
	import xgboost as xgb

	os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

	X_training_bis=X_training[:50000,features]
	y_training_bis=y_training[:50000]
	general_params = {"method": "XGB", "n_training": 50000, "n_validation": 5000, "selected_features": features}

	common_params = {}

	param_grid = {"silent":[True],
					  "scale_pos_weight":[1],
					  "objective":['binary:logistic'],
					  "reg_alpha" : [0.3],
					  "learning_rate":[0.01],
					  "colsample_bytree" :[1],
					  "subsample" : [0.95],
					  "n_estimators":list(range(100,1550,50)),
					  "max_depth":[3],
					  "gamma":[1]}
	gscv = GridSearchCV(xgb.XGBClassifier(),
				 param_grid,
				 cv=5,
				 n_jobs=6,
				 verbose=True)
	gscv.fit(X_training_bis,y=y_training_bis)
	param_grid["n_estimators"]=[gscv.best_params_["n_estimators"]]
	print("Optimal n_estimators : %d"%param_grid["n_estimators"][0])

	# for max_depth maximized it without overfitting
	param_grid["max_depth"] = list(range(3,11))
	scores = []
	for i,m_d in enumerate(param_grid["max_depth"]):
		this_param={}
		for key in param_grid:
			if key == "max_depth":
				this_param["max_depth"] = m_d
			else:
				print(param_grid[key])
				this_param[key]=param_grid[key][0]

		score = test(general_params,
					 this_param,
					 X_training,
					 y_training,
					 X_validation,
					 y_validation,
					 X_testing)
		scores.append(score)
	print(scores)
	# first index withing [max-treshold,max]
	max = max(scores)
	treshold = 0.002
	for i in range(len(scores)):
		if scores[i] > max - treshold:
			param_grid["max_depth"] = [list(range(3,11))[i]]
			break
	print("Max depth: %d"%param_grid["max_depth"][0])

	param_grid["learning_rate"] = [0.001,0.002,0.005,0.01]
	param_grid["subsample"] = list(np.arange(0.7,1,0.25))
	param_grid["colsample_bytree"] = list(np.arange(0.70,1,0.25))
	param_grid["gamma"] = [1,2,5]
	param_grid["objective"] = ["binary:logistic","binary:logitraw","binary:hinge"]

	gscv = GridSearchCV(xgb.XGBClassifier(),
						param_grid,
						cv=5,
						n_jobs=6,
						verbose=True)
	gscv.fit(X_training_bis, y=y_training_bis)
	print("Best params: %s"%str(gscv.best_params_))
	file = open("res_param.txt","w")
	for i in gscv.best_params_:
		file.write("%s: %s\n"%(str(i),str(gscv.best_params_[i])))
	file.close()
	"""