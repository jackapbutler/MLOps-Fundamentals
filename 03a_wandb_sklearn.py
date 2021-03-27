# Compare multiple models or perform hyperparameter tuning
# See results at wandb.ai (as a run in the user profile)  

# Packages
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes

from sklearn.model_selection import train_test_split

# Data
X = pd.read_csv('./data/processed_data.csv')
y = X.pop('Outcome') # ejects quality column as labels
features = X.columns

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Initialise some models
log = LogisticRegression(C=0.05, solver="lbfgs", max_iter=500)
svm = SVC(probability=True)
knn = KNeighborsClassifier(n_neighbors=10)
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=0.01)
labels = [0,1]

def model_algorithm(clf, X_train, y_train, X_test, y_test, name, labels, features):
    clf.fit(X_train, y_train)
    y_probas = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    wandb.init(project="sklearn_wandb", name=name, reinit=True)
    wandb.termlog('\nPlotting %s.'%name)

    wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
    wandb.termlog('Logged learning curve.')

    wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
    wandb.termlog('Logged confusion matrix.')

    wandb.sklearn.plot_summary_metrics(clf, X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    wandb.termlog('Logged summary metrics.')

    wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
    wandb.termlog('Logged class proportions.')
    
    if(not isinstance(clf, naive_bayes.MultinomialNB)):
        wandb.sklearn.plot_calibration_curve(clf, X_train, y_train, name)
    wandb.termlog('Logged calibration curve.')

    wandb.sklearn.plot_roc(y_test, y_probas, labels)
    wandb.termlog('Logged roc curve.')

    wandb.sklearn.plot_precision_recall(y_test, y_probas, labels)
    wandb.termlog('Logged precision recall curve.')

model_algorithm(log, X_train, y_train, X_test, y_test, 'LogisticRegression', labels, features)
model_algorithm(svm, X_train, y_train, X_test, y_test, 'SVM', labels, features)
model_algorithm(adaboost, X_train, y_train, X_test, y_test, 'AdaBoost', labels, features)
model_algorithm(knn, X_train, y_train, X_test, y_test, 'KNN', labels, features)
