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

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# Print the number of rows/people in the Enron dataset
print('Number of data points = ' + str(len(enron_data)))

# Print the number of unique features in the Enron dataset for each person
print('Number of features for each data point = ' 
      + str(len(enron_data.values()[0])))

# Print the person of interest count for the data set
poi_count = 0

for key in enron_data.keys():
    if enron_data[key].get('poi'):
        poi_count += 1
        
print('Number of people of interest in data set = ' + str(poi_count))