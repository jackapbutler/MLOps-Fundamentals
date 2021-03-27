# Packages
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

import json
import pickle 
import os
import yaml

# parameters
params = yaml.safe_load(open('./experiments/params.yaml'))['train_model']

# Data
X = pd.read_csv('./data/processed_data.csv')
y = X.pop('Outcome') # ejects quality column as labels

# Train / Test Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20)

# Fit model
layers = params['layers']
mlp = MLPClassifier(hidden_layer_sizes=layers)
mlp.fit(X_tr, y_tr)

# Make predictions
train_preds = mlp.predict(X_tr)
test_preds = mlp.predict(X_te)

# Results
train_score = accuracy_score(y_tr, train_preds)*100
test_score = accuracy_score(y_te, test_preds)*100

scores = {}
scores['Train accuracy'] = [train_score]
scores['Test accuracy'] = [test_score]
print(scores)
with open('./metrics/train_metrics.json', 'w') as outfile:
    json.dump(scores, outfile)

# Plot loss curve 
plt.plot(mlp.loss_curve_)
plt.title('Neural Network Layers: '+str(layers))
plt.savefig("./images/sklearn_mlp_loss_curve.png")
plt.close()

# Write the model to a file
if not os.path.isdir("models/"):
    os.mkdir("models")

filename = 'models/sklearn_neuralnet.pkl'
pickle.dump(mlp, open(filename, 'wb'))
