import vectorization as vect
import predict as pred
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

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
corpus_abstract = [element[5].split(" ") for element in node_info]
metas = [IDs,years,authors,corpus_abstract,corpus_title]



############################
# 1.2) METHODS TO GET PREDICTIONS USING VECTORIZATION METHODS FROM vectorization.py
#      AND PREDICTION METHODS FROM prediction.py


# SVM prediction
def get_predictions_svm():
    print("---- running...")
    list_sims1, list_sims2 = vect.compute_similarities(corpus_abstract,corpus_title)
    print("---- similarity matrices computed")
    metas = [IDs,list_sims1,list_sims2,years,authors]
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
    metas = [IDs,list_sims1,list_sims2,years,authors]
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
    # test a given set of params and save the output
    assert "method" in general_params.keys()
    n_training = general_params["n_training"]
    n_validation = general_params["n_validation"]
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
        model = pred.NNClassfier(X_training.shape[1],size_layers)
        model.fit(X_training,y_training,epochs=epochs)
    if general_params["method"] == "SVC":
        gamma = method_params["gamma"]
        selected_features = method_params["selected_features"]
        model = pred.SVMClassifier(gamma=gamma,features=selected_features)
        model.fit(X_training,y_training)
    # Test the model, saves a prediction for the testing set and log everything
    accuracy = model.accuracy(X_validation,y_validation)
    print(accuracy)
    pred_Y = model.predict(X_testing)
    pred_file = save_predictions(pred_Y)
    log_predictions(pred_file,method_params,general_params,accuracy)


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
    [11,"(Meta) Delta publication year",True],
    [12,"(Meta) Number of common authors",True],
    [13,"(Text) Abstract similitude",True],
    [14,"(Text) Title similitude",True]
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


if (__name__ == "__main__"):
    X_training = np.array(vect.get_features_of_set("training",metas))
    y_training = np.array([e[2] for e in training_set])
    X_validation = np.array(vect.get_features_of_set("validation",metas))
    y_validation = np.array([e[2] for e in validation_set])
    X_testing = np.array(vect.get_features_of_set("testing",metas))
    confront_features(X_training[:2000],y_training[:2000],11,12,"scatter_plot","Delta year vs. Common authors")
    confront_features(X_training[:2000],y_training[:2000],13,14,"scatter_plot","Abstract similitude vs. Title similitude")
    confront_features(X_training[:2000],y_training[:2000],3,4,"scatter_plot","Neighborhood distance vs. Neighbors measure")
    confront_features(X_training[:2000],y_training[:2000],2,8,"scatter_plot","Common neighbors vs. Total Neighborhood")

    """
    EXAMPLE FOR SVM
    general_params = {"method":"SVC","n_training":20000,"n_validation":3000}
    for gamma in np.arange(0.0005,0.005,0.0005):
        method_params = { "gamma" : gamma, "selected_features" : "all"}
        test(general_params,method_params,X_training,y_training,X_validation,y_validation,X_testing)
    """

    # parameters to

    # code to get and save the svm predictions.
    #Y = get_predictions_svm()
