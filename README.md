# The repository strucutre

The structure and implementation of the methods are conceivede to allow for complete modularity between each step of the pipeline : pre-processing, vectorization and prediction

## 1) vectorization.py

Groups all the vectorization methods and includes.
The meta-method "features.py" uses all available methods to produce a composite feature vector

## 2) predict.py 

Groups all the implemented prediction methods.
Every method should have a train and predict method with normalized entries : X_train, Y_train, X_test, Y_test
The logits classifier written by Jo does not conform yet, but we probably won't need it in the future?

## 3) main.py

Contains the pre-processing code as well as aggregate methods that rely on vectorization.py and predict.py to produce an experiment
Also contains a standard saving method which writes a Y_predict in the correct kaggle format (can somebody check that I used the right IDs ??)

# How to use

To produce a new experiment, write a new method in the 1.2 section of main.py and modify the __main__ loop at the end of main.py so that this new method is called (and that it's result is saved using the existing method)
Such methods such have a standard Y output ! 