import vectorization as vect
import predict as pred
import csv
from datetime import datetime


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

def get_predictions_svm():
    print("running")
    #vect.compute_similarities(corpus_abstract,corpus_title)
    #print(completed the computation and saving of similarity matrices)
    list_sims1, list_sims2 = vect.recover_list_sims()
    print("similarity matrices recovered")
    metas = [IDs,list_sims1,list_sims2,years,authors]
    X,Y = [],[]
    for id1,id2,y in training_set:
        Y.append(y)
        X.append(vect.features(id1,id2,metas))
    print("all parameters have been initialized")
    model = pred.train_svm(X,Y)
    print("svm model has been trained")
    X2 = testing_set
    Y2 = pred.predict_svm(model,X2)
    print("predictions have been made")
    return(Y2)


def save_predictions(Y):
    # saves a prediction list in the right format
    # assumes predictions have been made in the "natural" order (that of testing_set)
    name = "predictions_"+str(datetime.now().day)+"_"+str(datetime.now().month)+"_"+str(datetime.now().hour)+"h"+str(datetime.now().minute)
    with open("Data/Processed/"+name+".dat",'w') as out:
        out.write("id,category")
        for i in range(len(Y)):
            out.write(str(i)+","+str(Y[i]))
    print("prediction successfully written to Data/Processed"+name)

if __name__ == "__main__":
    Y = get_predictions_svm()
    save_predictions(Y)