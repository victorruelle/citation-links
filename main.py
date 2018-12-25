import vectorization as vect
import predict as pred
import csv
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

#here we open and process the testing set, ie pairs of articles without a link
with open("Data/testing_set.txt", "r") as f:
	reader = csv.reader(f)
	testing_set  = list(reader)
	
testing_set = [element[0].split(" ") for element in testing_set]
#we obtain a list of lists of 2 strings

#here we pre-process the training data
with open("Data/training_set.txt", "r") as f:
	reader = csv.reader(f)
	training_set  = list(reader)

training_set = [ [ int(sub_element) for sub_element in element[0].split(" ")] for element in training_set] # on converti directement en int!
#we obtain a list of lists of 3 strings: the two articles IDs and 1 if there is a link, 0 otherwise

with open("Data/node_information.csv", "r") as f:
	reader = csv.reader(f)
	node_info  = list(reader)


# create a list with list with all the words for each info
IDs = [int(element[0]) for element in node_info]
years = [int(element[1]) for element in node_info]
corpus_title = [element[2].split(" ") for element in node_info]
authors = [element[3].split(" ") for element in node_info]
corpus_abstract = [element[5].split(" ") for element in node_info]


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
    NN_pred = pred.create_NN(X_train,Y_train,testing_set)
    return(NN_pred)

############################
# 1.3) METHOD TO SAVE PREDICITONS IN THE RIGHT FORMAT

def save_predictions(Y):
    # saves a prediction list in the right format
    # assumes predictions have been made in the "natural" order (that of testing_set)
    name = "predictions_"+str(datetime.now().day)+"_"+str(datetime.now().month)+"_"+str(datetime.now().hour)+"h"+str(datetime.now().minute)
    with open("Data/Processed/"+name+".dat",'w') as out:
        out.write("id,category")
        for i in range(len(Y)):
            out.write(str(i)+","+str(Y[i]))
    print("prediction successfully written to Data/Processed"+name)


if (__name__ == "__main__"):
	Y = get_predictions_NN()
	save_predictions(Y)
    # code to get and save the svm predictions.
    #Y = get_predictions_svm()
    
