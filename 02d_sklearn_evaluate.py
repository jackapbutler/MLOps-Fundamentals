import sys
import os
import pickle
import json
import pandas as pd
import sklearn.metrics as metrics

# Files
model_file = './models/sklearn_neuralnet.pkl'
data_file = './data/processed_data.csv'
scores_file = './metrics/eval_score.json'
prc_file = './metrics/eval_prc.json'
roc_file = './metrics/eval_roc.json'

with open(model_file, 'rb') as fd:
    model = pickle.load(fd)

# Set Input and Labels
x = pd.read_csv(data_file)
labels = x.pop('Outcome') # ejects quality column as labels

# Make predictions
predictions_by_class = model.predict_proba(x)
predictions = predictions_by_class[:, 1]

# Accuracy metrics
precision, recall, prc_thresholds = metrics.precision_recall_curve(labels, predictions)
fpr, tpr, roc_thresholds = metrics.roc_curve(labels, predictions)

avg_prec = metrics.average_precision_score(labels, predictions)
roc_auc = metrics.roc_auc_score(labels, predictions)

# Store in files
with open(scores_file, 'w') as fd:
    json.dump({'avg_prec': avg_prec, 'roc_auc': roc_auc}, fd, indent=4)

with open(prc_file, 'w') as fd:
    json.dump({'prc': [{
            'precision': p,
            'recall': r,
            'threshold': t
        } for p, r, t in zip(precision, recall, prc_thresholds)
    ]}, fd, indent=4)

with open(roc_file, 'w') as fd:
    json.dump({'roc': [{
            'fpr': fp,
            'tpr': tp,
            'threshold': t
        } for fp, tp, t in zip(fpr, tpr, roc_thresholds)
    ]}, fd, indent=4)
