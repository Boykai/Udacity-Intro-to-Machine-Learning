#!/usr/bin/python

import sys
import pickle
from sklearn import metrics
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit
'''
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

1. Started by using ALL the feature for first iteration
2. After attempting to run feature_format, removed 'email_address' feature
   due to this feature throwing an error
'''

# Create feature list to include needed features for classifer
# 'poi' must be first feature within the list
# Features removed later in KBest and PCA pipeline
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

''' 
### Task 2: Remove outliers

1. Not removing ANY outliers for first iteration 
2. Remove 'TOTAL' from the dataset. It biases the dataset due to it being a 
   total of all the features for all of the samples
'''

# Print and remove 'TOTAL' from dataset
print('Removing "TOTAL"...' + str(data_dict['TOTAL']))
data_dict.pop('TOTAL', 0)

'''
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

1. Not adding ANY new features for first iteration
2. Create new feature, ratio of poi emails to total emails
'''

# Find ratio of poi emails to total emails
person_count = 0
mutated_data_dict = data_dict.copy()

for person in mutated_data_dict:
    ratio_poi_to_total_emails = 0.0
    person_features = mutated_data_dict[person]
    
    # Check value is int for email count features
    if isinstance(person_features['from_this_person_to_poi'], (int, long)) and \
       isinstance(person_features['from_poi_to_this_person'], (int, long)):
        total_poi_emails = float(person_features['from_this_person_to_poi']) \
                           + float(person_features['from_poi_to_this_person'])

        if total_poi_emails:
            total_emails = float(person_features['to_messages']) \
                           + float(person_features['from_messages'])
            ratio_poi_to_total_emails = total_poi_emails / total_emails
    
    person_count += 1
    person_features['poi email ratio'] = round(ratio_poi_to_total_emails, 5)

print('person count = ' + str(person_count))
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

'''
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

1. Created evaluateClf method in order to print out evaluation metrics
   for different ML classifers while keeping the code DRY
'''

# Method to print classifer elvaluation metrics
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
    roc_auc = metrics.roc_auc_score(labs_test, predictions)
    
    print('\n' + str(type(classifer)))
    print('Accuracy = ' + str(accuracy))
    print('Percision = ' + str(precision))
    print('Recall = ' + str(recall))
    print('F1 Score = ' + str(f1_score))
    print('ROC Curve AUC = ' + str(roc_auc))

# Provided to give you a starting point. Try a variety of classifiers.
'''
1. Created an basic instance of some classifer models
'''
# Split data into training and testing sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Import classifer model libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# Create list of basic classifers
classifiers = [
    KNeighborsClassifier(2),
    SVC(),
    DecisionTreeClassifier(),
    #RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB()]
    
# Interate over each basic model to see which ones perform best
for model in classifiers:
    clf = model
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    evaluateClf(clf, features_test, labels_test, pred)


'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
'''

# Create parameter grid options for each classifer, store in params_list
params_list = []
feature_params_list = dict(reduce_dim__n_components = np.arange(1, 4),
                           reduce_dim__whiten = [True, False],
                           reduce_dim__svd_solver = ['auto', 'full', 'arpack', 'randomized'])

# Create cross validation metric
print('Calculating cross valadation...')
cv = StratifiedShuffleSplit(labels_train, 10, random_state = 42)


# KNeighbors parameters for GridSearchCV
kneighbors_params = dict(clf__metric = ['minkowski','euclidean','manhattan'], 
                         clf__weights = ['uniform', 'distance'],
                         clf__n_neighbors = np.arange(2, 10),
                         clf__algorithm = ['auto', 'ball_tree', 'kd_tree','brute'])
kneighbors_params.update(feature_params_list)
params_list.append(kneighbors_params)

# SVM parameters for GridSearchCV
svc_params = dict(clf__C = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000],
                      clf__gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                      clf__kernel= ['rbf'], 
                      clf__class_weight = ['balanced', None],
                      clf__random_state = [42])
svc_params.update(feature_params_list)
params_list.append(svc_params)

# Decision Tree parameters for GridSearchCV
decision_tree_params = dict(clf__criterion = ['gini', 'entropy'],
                            clf__max_features = ['sqrt', 'log2', None],
                            clf__class_weight = ['balanced', None],
                            clf__random_state = [42])
decision_tree_params.update(feature_params_list)
params_list.append(decision_tree_params)

# Random Forest parameters for GridSearchCV
random_forest_params = dict(clf__n_estimators = np.arange(10, 50, 10),
                             clf__criterion = ['gini', 'entropy'],
                             clf__max_features = ['sqrt', 'log2', None],
                             clf__class_weight = ['balanced', None],
                             clf__random_state = [42])
random_forest_params.update(feature_params_list)
params_list.append(random_forest_params)

# Adaboost parameters for GridSearchCV
adaboost_params = dict(clf__base_estimator = [DecisionTreeClassifier(),
                                              GaussianNB()],
                       clf__n_estimators = np.arange(10, 150, 10),
                       clf__algorithm = ['SAMME', 'SAMME.R'],
                       clf__random_state = [42])
adaboost_params.update(feature_params_list)
params_list.append(adaboost_params)

# Naive Bayes parameters for GridSearchCV
naive_bayes_params = dict()
naive_bayes_params.update(feature_params_list)
params_list.append(naive_bayes_params)

# Iterate over each classifier and their parameters, apply PCA and GridsearchCV
best_estimators = {}

for i in range(len(params_list)):
    print('\nCalculating scaled features, classifier parameters, and PCA...')
    print(str(type(classifiers[i])))
    
    # Create pipeline and apply GridSearchCV
    estimators = [('scalar', preprocessing.MinMaxScaler()),
                  ('selector', SelectKBest()),
                  ('reduce_dim', PCA()), 
                  ('clf', classifiers[i])]
    pipe = Pipeline(estimators) 
    grid = GridSearchCV(pipe, 
                        param_grid = params_list[i], 
                        scoring = 'f1',
                        cv = cv)
    
    try:
        grid.fit(features_train, labels_train)
    except:
        grid.fit(np.array(features_train), np.array(labels_train))

    pred = grid.best_estimator_.predict(features_test)
    
    f1_score = metrics.f1_score(labels_test, pred)

    # Evaluate the best estimator
    evaluateClf(grid.best_estimator_, features_test, labels_test, pred)
    
    # Get features used in best estimator
    #print('The features used are: \n' + str(grid.best_estimator_.best_params['selector__k']))
    
    # Run test_classifer
    print('\n\nRunning Tester...\n' + str(type(classifiers[i])))
    test_classifier(grid.best_estimator_, my_dataset, features_list)

    best_estimators.update({f1_score : grid.best_estimator_})
    
    print('\nBest estimator = \n' + str(grid.best_estimator_))

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
'''
# Final classifer to be used
svc_params = dict(clf__C = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000],
                      clf__gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                      clf__kernel= ['rbf'], 
                      clf__class_weight = ['balanced', None],
                      clf__random_state = [42])
svc_params.update(feature_params_list)
params_list.append(svc_params)

print('\nCalculating cross validation, scaled features, classifier parameters, and PCA...')
    
# Create pipeline and apply GridSearchCV
print('Calculating estimators...')
estimators = [('scalar', preprocessing.MinMaxScaler()),
              ('selector', SelectKBest()),
              ('reduce_dim', PCA()), 
              ('clf', SVC())]

print('Creating pipeline...')
pipe = Pipeline(estimators) 
print('Calculating grid search...')
grid = GridSearchCV(pipe, 
                    param_grid = params_list[0], 
                    scoring = 'f1',
                    cv = cv)

try:
    grid.fit(features_train, labels_train)
except:
    grid.fit(np.array(features_train), np.array(labels_train))

pred = grid.best_estimator_.predict(features_test)
evaluateClf(grid.best_estimator_, features_test, labels_test, pred)

clf = grid.best_estimator_

print('Calculations finished.')
#clf = best_estimators[best_f1]
'''
'''
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
'''
dump_classifier_and_data(clf, my_dataset, features_list)