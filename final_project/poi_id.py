#!/usr/bin/python

import sys
import pickle
from sklearn import metrics
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from tester import dump_classifier_and_data
from tester import test_classifier
# Import classifer model libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
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
def getFeatureList():
    '''
    Creates list of labels for features of Enron dataset
    
    @return: features_list (a list)
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
    return features_list

def getDataDict():
    '''
    Get the dictonary containing the dataset from pickle file.
    
    data_dict contains keys of people in Eron, with values of dictonaries
    with each feature being a key
    with each feature value being a value
    
    @return: data_dict (a dict)
    '''    
    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    
    return data_dict

def removeOutliers(data_dict):
    ''' 
    Remove bad outliers from Enron dataset
    Removes 'TOTAL' outlier entry from data_dict
    Returns clean dataset with outliers are removed
    
    data_dict: Dictonary of Enron dataset (a dict)
    
    @return: data_dict (a dict)
    '''
    ### Task 2: Remove outliers
    # 1. Not removing ANY outliers for first iteration 
    # 2. Remove 'TOTAL' from the dataset. It biases the dataset due to it being a 
    # total of all the features for all of the samples
       
    # Print and remove 'TOTAL' from dataset
    print('Removing "TOTAL"...\n' + str(data_dict['TOTAL']))
    data_dict.pop('TOTAL', 0)
    
    return data_dict

def createFeatures(data_dict):
    '''
    Creates new feature and updates dataset dict (data_dict)
    Returns updated dataset dict with new feature added
    
    @return: data_dict (a dict)
    '''
    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    # 1. Not adding ANY new features for first iteration
    # 2. Create new feature, ratio of poi emails to total emails
    
    # Find ratio of poi emails to total emails
    mutated_data_dict = data_dict.copy()
    
    # Iterate over each person in dataset, get required feature values
    for person in mutated_data_dict:
        ratio_poi_to_total_emails = 0.0
        person_features = mutated_data_dict[person]
        
        # Check value is int for email count features
        if isinstance(person_features['from_this_person_to_poi'], (int, long)) and \
           isinstance(person_features['from_poi_to_this_person'], (int, long)):
            total_poi_emails = float(person_features['from_this_person_to_poi']) \
                             + float(person_features['from_poi_to_this_person'])
            # Check total_poi_emails is not NULL
            if total_poi_emails:
                total_emails = float(person_features['to_messages']) \
                             + float(person_features['from_messages'])
                # Calculate total poi emails to total emails
                ratio_poi_to_total_emails = total_poi_emails / total_emails
        # Create, store, and update new feature 'poi_emails_ratio'
        person_features['poi_email_ratio'] = round(ratio_poi_to_total_emails, 5)
    
    my_dataset = mutated_data_dict
    
    return my_dataset

# Method to print classifer elvaluation metrics
def evaluateClf(classifer, feats_test, labs_test, predictions):
    '''
    Evaluates ML classifer using different metrics, such as:
        Accuracy
        Precision
        Recall
        F1 Score
        ROC Curve AUC
        
    classifer: ML classifer model object (an object)
        
    feats_test: List of feature values within the test subset (a list)

    labs_test: List of label values within the test subset (a list)
        
    prediction: List of prediction label values based on the test subset
                (a list)
    '''
    # 1. Created evaluateClf method in order to print out evaluation metrics
    
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

def simpleClassifiers(classifiers, features_train, labels_train, 
                      features_test, labels_test):    
    '''
    Runs and evaluates multiple simple ML classifiers on Enron dataset then,
    reports the resuts of the classifier evaluation
    
    classifiers: List of classifier objects to fit, predict, and evaluate on
                 the Enron dataset (a list)
    
    features_train: List of features to train the classifier (a list)
    
    labels_train: List of labels to train the classifier (a list)
    
    features_test: List of features to test the classifier (a list)
    
    labels_test: List of labels to test the classifier (a list)
    '''
    ### Task 4: Try a varity of classifiers    
    # Provided to give you a starting point. Try a variety of classifiers.
    # 1. Created an basic instance of some classifer models
    
    # Interate over each basic model to see which ones perform best
    for model in classifiers:
        clf = model
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        evaluateClf(clf, features_test, labels_test, pred)

        
def getPCAKBestParameters(features):
    '''
    Creates parameter dict for PCA and SelectKBest for later use in parameter
    tuning in GridSearchCV
    
    reduce_dim__n_components must be strickly less than min(selector__k)
    
    selector__k must be strickly greater than max(reduce_dim__n_components)
    
    @return: PCA and KBest parameter list (a dict)
    '''
    
    feature_params_list = dict(reduce_dim__n_components = np.arange(1, 4),
                               reduce_dim__whiten = [True, 
                                                     False],
                               reduce_dim__svd_solver = ['auto', 
                                                         'full', 
                                                         'arpack', 
                                                         'randomized'],
                               selector__k = np.arange(5, len(features) - 1))  
    
    return feature_params_list
    
def getParameters(classifiers, features_list):
    '''
    Creates parameter list for each classifier for later use in parameter
    tuning in GridSearchCV
    
    classifiers: List of classifier objects to fit, predict, and evaluate on
                 the Enron dataset (a list)
                 
    features_list: List of labels for features of Enron dataset (a list)
    
    @return: Classifier key, parameter list, pairs for GridSearchCV use (a dict)
    '''
    
    # Create parameter grid options for each classifer, store in params_list
    params_list = []
    
    # Get PCA and SelectKBest parameter list for GridSearchCV
    feature_params_list = getPCAKBestParameters(features_list)   
    
    # KNeighbors parameters for GridSearchCV
    kneighbors_params = dict(clf__metric = ['minkowski','euclidean','manhattan'], 
                             clf__weights = ['uniform', 'distance'],
                             clf__n_neighbors = np.arange(2, 10),
                             clf__algorithm = ['auto', 'ball_tree', 'kd_tree','brute'])
    kneighbors_params.update(feature_params_list)
    params_list.append(kneighbors_params)
    
    # SVM parameters for GridSearchCV
    svc_params = dict(clf__C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 10, 100, 1000, 10000],
                          clf__gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                          clf__kernel= ['rbf'], 
                          clf__class_weight = ['balanced', None],
                          clf__random_state = [0, 1, 10, 42])
    svc_params.update(feature_params_list)
    params_list.append(svc_params)
    
    # Decision Tree parameters for GridSearchCV
    decision_tree_params = dict(clf__criterion = ['gini', 'entropy'],
                                clf__max_features = ['sqrt', 'log2', None],
                                clf__class_weight = ['balanced', None],
                                clf__random_state = [0, 1, 10, 42])
    decision_tree_params.update(feature_params_list)
    params_list.append(decision_tree_params)
    
    # Random Forest parameters for GridSearchCV
    random_forest_params = dict(clf__n_estimators = np.arange(10, 50, 10),
                                 clf__criterion = ['gini', 'entropy'],
                                 clf__max_features = ['sqrt', 'log2', None],
                                 clf__class_weight = ['balanced', None],
                                 clf__random_state = [0, 1, 10, 42])
    random_forest_params.update(feature_params_list)
    params_list.append(random_forest_params)
    
    # Adaboost parameters for GridSearchCV
    adaboost_params = dict(clf__base_estimator = [DecisionTreeClassifier(),
                                                  GaussianNB()],
                           clf__n_estimators = np.arange(10, 150, 10),
                           clf__algorithm = ['SAMME', 'SAMME.R'],
                           clf__random_state = [0, 1, 10, 42])
    adaboost_params.update(feature_params_list)
    params_list.append(adaboost_params)
    
    # Naive Bayes parameters for GridSearchCV
    naive_bayes_params = dict()
    naive_bayes_params.update(feature_params_list)
    params_list.append(naive_bayes_params)
    
    classifiers_params_dict = {}
    
    for i in classifiers:
        classifiers_params_dict.update({classifiers[i] : params_list[i]})
        
    return classifiers_params_dict
        

    
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

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
    # https://discussions.udacity.com/t/how-to-find-out-the-features-selected-by-selectkbest/45118/4
    features_selected_bool = grid.best_estimator_.named_steps['selector'].get_support()
    features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
    print('The features used are: \n' + str(features_selected_list))
    
    # Run test_classifer
    print('\n\nRunning Tester...\n' + str(type(classifiers[i])))
    test_classifier(grid.best_estimator_, my_dataset, features_list)

    best_estimators.update({f1_score : grid.best_estimator_})
    
    print('\nBest estimator = \n' + str(grid.best_estimator_))
# END OF TUNING MULTIPLE CLASSIFIER PIPELINE TYPES


# START OF FINAL TUNED CLASSIFIER
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Final classifer to be used
# Naive Bayes parameters for GridSearchCV
final_params_list = []
final_feature_params_list = dict(reduce_dim__n_components = np.arange(1, 4),
                           reduce_dim__whiten = [True, False],
                           reduce_dim__svd_solver = ['auto', 'full', 'arpack', 'randomized'],
                           selector__k = [5, 10, 15, 'all'])
final_naive_bayes_params = dict()
final_naive_bayes_params.update(final_feature_params_list)
final_params_list.append(final_naive_bayes_params)

print('\nCalculating cross validation, scaled features, classifier parameters, and PCA...')
    
# Create pipeline and apply GridSearchCV
print('Calculating estimators...')
final_estimators = [('scalar', preprocessing.MinMaxScaler()),
                    ('selector', SelectKBest()),
                    ('reduce_dim', PCA()), 
                    ('clf', GaussianNB())]

print('Creating pipeline...')
final_pipe = Pipeline(final_estimators) 

print('Creating cross validation function...')
final_cv = StratifiedShuffleSplit(labels_train, 10, random_state = 42)

print('Calculating grid search...')
grid = GridSearchCV(final_pipe, 
                    param_grid = final_params_list, 
                    scoring = 'f1',
                    cv = final_cv)

print('Fitting classifier to dataset...')
try:
    grid.fit(features_train, labels_train)
except:
    grid.fit(np.array(features_train), np.array(labels_train))

print('Getting predictions for classifier of dataset...')
final_pred = grid.best_estimator_.predict(features_test)

print('Printing evaluation metrics for classifier on testing subset of dataset...')
evaluateClf(grid.best_estimator_, features_test, labels_test, final_pred)

# Get features used in best estimator
# https://discussions.udacity.com/t/how-to-find-out-the-features-selected-by-selectkbest/45118/4
features_selected_bool = grid.best_estimator_.named_steps['selector'].get_support()
features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
features_scores = ['%.2f' % elem for elem in grid.best_estimator_.named_steps['selector'].scores_]
features_selected_scores = [x for x, y in zip(features_scores, features_selected_bool) if y]

print('\nThe features used are:')

for i in range(len(features_selected_list)):
    print(str(features_selected_list[i]) + ' ' + str(features_selected_scores[i]))

# Set clf, pipeline object passed into tester.py for evaluation by grader
clf = grid.best_estimator_

print('\nCalculations finished.')
# END OF FINAL TUNED CLASSIFIER

'''
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
'''
dump_classifier_and_data(clf, my_dataset, features_list)


def main():
    # Get, create, and store Enron dataset
    feature_names = getFeatureList()
    dataset = getDataDict() 
    dataset = removeOutliers(dataset)
    #dataset = createFeatures(dataset) # Uncomment to add new features
    
    # Extract features and labels from dataset for local testing
    data = featureFormat(dataset, feature_names, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    # Split data into training and testing sets
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Create list of basic classifers
    classifiers = [KNeighborsClassifier(),
                   SVC(),
                   DecisionTreeClassifier(),
                   RandomForestClassifier(),
                   AdaBoostClassifier(),
                   GaussianNB()]
                   
    # Evaluate basic classifiers with no parameters on Enron dataset
    simpleClassifiers(classifiers, features_train, labels_train,
                      features_test, labels_test)
    
    # Get dictonary of classifier, parameter, pairs
    classifiers_params_list = getParameters(classifiers, feature_names)
    
    
    # Create cross validation metric
    print('Calculating cross valadation...')
    cv = StratifiedShuffleSplit(labels_train, 10, random_state = 42)
    
if __name__ == '__main__':
    main()
