#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
"""
from collections import Counter
import pickle

enron_data = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))

number_of_pois = len(
    Counter(person for person in enron_data if enron_data[person]["poi"]))
print len(
    Counter(person for person in enron_data
            if enron_data[person]["total_payments"] == "NaN"))
print number_of_pois
