# Because they're run from DVC these test files actually are in the previous directory!

import pickle
from sklearn.datasets import make_classification
import json

model_file = './models/sklearn_neuralnet.pkl'
model = pickle.load(open(model_file, "rb"))

# Generate some data for validation
X_test, y = make_classification(n_samples= 1000, n_features= 8, n_classes= 2) # n_features excluding labels 

# Test that the model can predict
y_hat = model.predict(X_test)