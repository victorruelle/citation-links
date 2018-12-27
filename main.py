import vectorization as vect
import predict as pred
import csv
import numpy as np
import os
from datetime import datetime

'''
MAIN FILE

1. CONTAINS : 
    1.1) CODE TO READ THE DATA AND PLACE IT IN USABLE LISTS
    1.1) METHODS TO GET PREDICTIONS USING VECTORIZATION METHODS FROM vectorization.py
    AND PREDICTION METHODS FROM prediction.py
    1.2) METHOD TO SAVE A PREDICTION IN THE RIGHT FORMAT

2. WHEN main.py IS CALLED :
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

def log_predictions(file_name,params,accuracy):
    log_file = "Predictions/log_predictions.txt"
    headers = "accuracy,prediction_file,params\n"
    if not os.path.isfile(log_file):
        open(log_file,"w").write(headers)
    file = open(log_file,"a")
    file.write("%f;%s;%s\n" % (accuracy,file_name,str(params)))

def test(params,X_training,y_training,X_validation,y_validation,X_testing):
    # test a given set of params and save the output
    assert "method" in params.keys()
    n_training = params["n_training"]
    n_validation = params["n_validation"]
    # takes random part of the data
    permutation_training = np.random.permutation(len(X_training))[:n_training]
    permutation_validation = np.random.permutation(len(X_validation))[:n_validation]
    X_training = X_training[permutation_training,:]
    y_training = y_training[permutation_training]
    X_validation = X_validation[permutation_validation,:]
    y_validation = y_validation[permutation_validation]
    print(params)
    if params["method"] == "NN":
        size_layers = params["size_layers"]
        epochs = params["epochs"]
        model = pred.NNClassfier(X_training.shape[1],size_layers)
        model.fit(X_training,y_training,epochs=epochs)
    accuracy = model.accuracy(X_validation,y_validation)
    print(accuracy)
    pred_Y = model.predict(X_testing)
    pred_file = save_predictions(pred_Y)
    log_predictions(pred_file,params,accuracy)


if (__name__ == "__main__"):
    X_training = np.array(vect.get_features_of_set("training",metas))
    y_training = np.array([e[2] for e in training_set])
    X_validation = np.array(vect.get_features_of_set("validation",metas))
    y_validation = np.array([e[2] for e in validation_set])
    X_testing = np.array(vect.get_features_of_set("testing",metas))
    params = {"method":"NN","n_training":100000,"n_validation":5000,"epochs":1}
    for n_1 in range(2,20,5):
        for n_2 in range(2,20,5):
            params["size_layers"] = [n_1,n_2]
            test(params,X_training,y_training,X_validation,y_validation,X_testing)

    # parameters to

    # code to get and save the svm predictions.
    #Y = get_predictions_svm()
