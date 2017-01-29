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
import pandas as pd

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
        
print('Total number of people of interest in data set = ' + str(poi_count))


# Print the total value of the stock belonging to James Prentice
print('Total number value of the stock belonging to James Prentice = '
      + str(enron_data['PRENTICE JAMES']['total_stock_value']))

# Print the value of stock options exercised by Jeff Skilling
print('Value of stock options exercised by Jeffrey Skilling = '
      + str(enron_data['SKILLING JEFFREY K']['exercised_stock_options']))


# Print "Of these three individuals (Lay, Skilling, and Fastow), who took home
# the most money (largest value of "total_payments" feature)? How much money
# did that person get?"
lay_take_home_pay = enron_data['LAY KENNETH L']['total_payments']
print('Lay took home $'
      + str(lay_take_home_pay))

skilling_take_home_pay = enron_data['SKILLING JEFFREY K']['total_payments']

print('Skilling took home $'
      + str(skilling_take_home_pay))

fastow_take_home_pay = enron_data['FASTOW ANDREW S']['total_payments']
print('Fastow took home $'
      + str(fastow_take_home_pay))

print('Lay received the most money at $'
      + str(lay_take_home_pay))


# Print how many folks in the dataset have a quantified salary
enron_df = pd.DataFrame(enron_data).T

enron_rows = len(enron_df)
print('Number of non-NaN salaries = '
      + str(enron_rows - enron_df['salary'].isin(['NaN']).sum()))

# Print how many folk in the dataset have email addresses
print('Number of non-NaN email addresses = '
      + str(enron_rows - enron_df['email_address'].isin(['NaN']).sum()))


# Print the percentage of people in the dataset that have "NaN" for their
# total payments
print('Percentage of people that have NaN for their total payments = '
      + str(float(enron_df['total_payments'].isin(['NaN']).sum())/float(enron_rows) * 100.0)
      + '%')
