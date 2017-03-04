#!/usr/bin/python

import sys
import pickle
from sklearn import metrics
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

''' 
1. Started by using ALL the feature for first iteration
2. After attempting to run feature_format, removed 'email_address' feature
   due to this feature throwing an error
'''
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments',
                 'total_payments', 'exercised_stock_options', 'bonus',
                 'restricted_stock', 'shared_receipt_with_poi',
                 'restricted_stock_deferred', 'total_stock_value',
                 'expenses', 'loan_advances', 'from_messages', 'other',
                 'from_this_person_to_poi', 'director_fees', 'deferred_income',
                 'long_term_incentive', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
'''
1. Not removing ANY outliers for first iteration 
2. Remove 'TOTAL' from the dataset. It biases the dataset due to it being a 
   total of all the features for all of the samples
2. Define method for removing a percentage of various variable outliers 
   after exploring the data visually and through regression
'''
# Print and remove 'TOTAL' from dataset
print(data_dict['TOTAL'])
data_dict.pop('TOTAL', 0)

def outlierCleaner(predictions, var1, var2):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual var2).

        @Return: a list of tuples named cleaned_data where 
        each tuple is of the form (var1, var2, error).
    """
    
    cleaned_data = []
    
    # Get the errors of predictions to net_worth
    errors = abs(predictions - var2)

    sorted_errors = sorted(errors)

    # Get the lowest 90% of errors, cut off highest 10% of errors
    percent_of_errors = sorted_errors[ : int(len(sorted_errors) * 0.9)]

    # Store all age, net_worth, error values in list if error is in lower 90%
    for i in range(len(errors)):
        if errors[i] <= percent_of_errors[-1]:
                cleaned_data.append((var1[i], var2[i], errors[i]))
    
    return cleaned_data


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
'''
1. Not adding ANY new features for first iteration
'''

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
'''
1. Created evaluateClf method in order to print out evaluation metrics
   for different ML classifers while keeping the code DRY
'''
def evaluateClf(classifer, feats_test, labs_test, predictions):
    '''
    Evaluates ML classifer using different metrics, such as:
        Accuracy
        Precision
        Recall
        F1 Score
        
        classifer: ML classifer model object (an object)
        
        feats_test: List of feature values within the test subset (a list)

        labs_test: List of label values within the test subset (a list)
        
        prediction: List of prediction label values based on the test subset
                    (a list)
    '''
    accuracy = classifer.score(feats_test, labs_test)
    precision = metrics.precision_score(labs_test, predictions)
    recall = metrics.recall_score(labs_test, predictions)
    f1_score = metrics.f1_score(labs_test, predictions)
    
    print('\n' + str(type(classifer)))
    print('Accuracy = ' + str(accuracy))
    print('Percision = ' + str(precision))
    print('Recall = ' + str(recall))
    print('F1 Score = ' + str(f1_score))


# Provided to give you a starting point. Try a variety of classifiers.
'''
1. Created an basic instance of some classifer models
'''
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)



# Import classifer model libraries
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Create list of basic classifers
classifiers = [
    KNeighborsClassifier(2),
    SVC(),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
    
# Interate over each basic model to see which ones perform best
for model in classifiers:
    clf = model
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    evaluateClf(clf, features_test, labels_test, pred)

    
clf = AdaBoostClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
evaluateClf(clf, features_test, labels_test, pred)    
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)