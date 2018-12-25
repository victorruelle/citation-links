from keras.models import Sequential
from keras.layers import Dense
import numpy

'''
THIS FILE HAS BEEN INCORPORATED IN PREDICTION.PY, WILL BE DELETED AFTER CONFIRMATION
'''

def create_NN(X_train, Y_train, X_test, Y_test):
	# fix random seed for reproducibility
	numpy.random.seed(7)

	# create model
	model = Sequential()
	#change numbers and number of features
	#Dense is fully connected, maybe make it half connected?
	#change number of layers
	model.add(Dense(12, input_dim=4, activation='relu'))
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