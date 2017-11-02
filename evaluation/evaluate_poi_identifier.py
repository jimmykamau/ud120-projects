#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score, confusion_matrix)
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))

# add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# your code goes here
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print accuracy_score(pred, labels_test)
print classification_report(labels_test, pred)
print confusion_matrix(labels_test, pred)
print len(features_test)
print "Accuracy if 0: {}".format(accuracy_score([0.] * len(pred), labels_test))
print "True positives: {}".format(
    len([i for i, j in zip(labels_test, pred) if i == j == 1]))
print "Precision score: {}".format(precision_score(labels_test, pred))
print "Recall score: {}".format(recall_score(labels_test, pred))
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print "Test precision score: {}".format(precision_score(true_labels, predictions))
print "Test recall score: {}".format(recall_score(true_labels, predictions))
