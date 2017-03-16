#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
sys.path.append("../tools/")
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from tester import test_classifier
from tester import dump_classifier_and_data
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def getFeatureList():
    '''
    Creates list of labels for features of Enron dataset
    
    @return: features_list (a list)
    '''
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    # 1. Started by using ALL the feature for first iteration
    # 2. After attempting to run feature_format, removed 'email_address' feature
    # due to this feature throwing an error
    
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
    print('\nRemoving "TOTAL"...\n' + str(data_dict['TOTAL']))
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

    
def evaluateDataset(data_dict):
    '''
    Prints basic Enron dataset People of Interest numerical values
    
    data_dict: Dictonary of Enron dataset (a dict)
    '''
    data = data_dict.copy()
    
    # Create pandas dataframe from Enron dataset
    df = pd.DataFrame(data).transpose()
    
    # Get people and feature count
    people_count, feature_count = df.shape
    people_poi_count = df['poi'].count()
    
    # Print basic Enron dataset People of Interest numerical values
    print('Enron dataset contains ' 
          + str(people_count) + ' people.')
    
    print('Enron dataset contains ' 
          + str(df['poi'].sum()) + ' People of Interest.')
    
    print('Enron dataset contains ' 
          + str(people_poi_count - df['poi'].sum()) + ' Non-People of Interest.')
    
       
def evaluateClf(classifer, feats_test, labs_test, predictions):
    '''
    Prints classifer elvaluation metrics
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
                               selector__k = [5, 10, 15, 'all'])  
    
    return feature_params_list

    
def getKNeighborsParams():
    '''
    Get list of values for each parameter for KNightbors classifier
    
    @return: dictionary containing list of values for each parameter (a dict)
    '''
    kneighbors_params = dict(clf__metric = ['minkowski',
                                            'euclidean',
                                            'manhattan'], 
                             clf__weights = ['uniform', 
                                             'distance'],
                             clf__n_neighbors = np.arange(2, 10),
                             clf__algorithm = ['auto', 
                                               'ball_tree', 
                                               'kd_tree',
                                               'brute'])
    
    return kneighbors_params
    
    
def getSVCParams():
    '''
    Get list of values for each parameter for SVC classifier
    
    @return: dictionary containing list of values for each parameter (a dict)
    '''
    svc_params = dict(clf__C = [0.00001, 
                                0.0001, 
                                0.001, 
                                0.01, 
                                0.1, 
                                10, 
                                100, 
                                1000, 
                                10000],
                          clf__gamma = [0.0001, 
                                        0.0005, 
                                        0.001, 
                                        0.005, 
                                        0.01, 
                                        0.1],
                          clf__kernel= ['rbf'], 
                          clf__class_weight = ['balanced', 
                                               None],
                          clf__random_state = [0, 
                                               1, 
                                               10, 
                                               42])
    
    return svc_params
    
    
def getDTParams():
    '''
    Get list of values for each parameter for Decision Tree classifier
    
    @return: dictionary containing list of values for each parameter (a dict)
    '''
    decision_tree_params = dict(clf__criterion = ['gini', 
                                                  'entropy'],
                                clf__max_features = ['sqrt', 
                                                     'log2', 
                                                     None],
                                clf__class_weight = ['balanced', 
                                                     None],
                                clf__random_state = [0, 
                                                     1, 
                                                     10, 
                                                     42])
    
    return decision_tree_params    

    
def getRandomForestParams():
    '''
    Get list of values for each parameter for Random Forest classifier
    
    @return: dictionary containing list of values for each parameter (a dict)
    '''
    random_forest_params = dict(clf__n_estimators = np.arange(10, 50, 10),
                                 clf__criterion = ['gini', 
                                                   'entropy'],
                                 clf__max_features = ['sqrt', 
                                                      'log2', 
                                                      None],
                                 clf__class_weight = ['balanced', 
                                                      None],
                                 clf__random_state = [0, 
                                                      1, 
                                                      10, 
                                                      42])
    
    return random_forest_params

    
def getAdaParams():
    '''
    Get list of values for each parameter for Adaboost classifier
    
    @return: dictionary containing list of values for each parameter (a dict)
    '''
    adaboost_params = dict(clf__base_estimator = [DecisionTreeClassifier(),
                                                  GaussianNB()],
                           clf__n_estimators = np.arange(10, 150, 10),
                           clf__algorithm = ['SAMME', 
                                             'SAMME.R'],
                           clf__random_state = [0, 
                                                1, 
                                                10, 
                                                42]) 
    
    return adaboost_params
    
    
def getParameters(classifiers, features_list):
    '''
    Creates parameter list for each classifier for later use in parameter
    tuning in GridSearchCV
    
    classifiers: List of classifier objects to fit, predict, and evaluate 
                 on the Enron dataset (a list)
                 
    features_list: List of labels for features of Enron dataset (a list)
    
    @return: Classifier type key, parameter dict, pairs for GridSearchCV use (a dict)
    '''
    
    # Create parameter grid options for each classifer, store in params_list
    param_dict = {}
    
    # Get PCA and SelectKBest parameter list for GridSearchCV
    feature_params_list = getPCAKBestParameters(features_list)   
    
    # KNeighbors parameters for GridSearchCV
    kneighbors_params = getKNeighborsParams()
    kneighbors_params.update(feature_params_list)
    param_dict.update({type(KNeighborsClassifier()) : kneighbors_params})
    
    # SVM parameters for GridSearchCV
    svc_params = getSVCParams()
    svc_params.update(feature_params_list)
    param_dict.update({type(SVC()) : svc_params})
    
    # Decision Tree parameters for GridSearchCV
    decision_tree_params = getDTParams()
    decision_tree_params.update(feature_params_list)
    param_dict.update({type(DecisionTreeClassifier()) : decision_tree_params})
    
    # Random Forest parameters for GridSearchCV
    random_forest_params = getRandomForestParams()
    random_forest_params.update(feature_params_list)
    param_dict.update({type(RandomForestClassifier()) : random_forest_params})
    
    # Adaboost parameters for GridSearchCV
    adaboost_params = getAdaParams()
    adaboost_params.update(feature_params_list)
    param_dict.update({type(AdaBoostClassifier()) : adaboost_params})
    
    # Naive Bayes parameters for GridSearchCV
    naive_bayes_params = dict()
    naive_bayes_params.update(feature_params_list)
    param_dict.update({type(GaussianNB()) : naive_bayes_params})
    
    return param_dict    
    
    
def getClassiferTunes(classifiers, params_list):
    '''
    Matches the classifier objects with the correct parameter list dictionaries
    to later be used in GridSearchCV
    
    classifiers: List of classifier objects to fit, predict, and evaluate 
                 on the Enron dataset (a list)
        
    params_list: Classifier type key, parameter dict, pairs for GridSearchCV use (a dict)
    
    @return: Dictonary classifier object and matching parameters (a dict)
    '''
    classifiersToTune = {}
    for classifier in classifiers:
        classifiersToTune.update({classifier : params_list[type(classifier)]})
     
    return classifiersToTune
    

def tuneClassifier(classifiers_params, cross_val, my_dataset, features_list,
                   features_train, labels_train, features_test, labels_test):
    '''
    Tune given classifiers with given matching params, then return the best 
    estimator given the classifiers input
    
    classifiers: List of classifier objects to fit, predict, and evaluate on
                 the Enron dataset (a list)
    
    my_dataset: Dict of people in Eron, with values of dictonaries (a dict)
    
    cross_val: Cross validation method object (a method object)
    
    features_list: list of labels for features of Enron dataset (a list)             
                 
    features_train: List of features to train the classifier (a list)
    
    labels_train: List of labels to train the classifier (a list)
    
    features_test: List of features to test the classifier (a list)
    
    labels_test: List of labels to test the classifier (a list)
    
    @return: Best classifier given the scoring metric (a pipeline object)
    '''    
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    # Iterate over each classifier and their parameters, apply PCA and GridsearchCV
    best_estimators = {}
    
    # Split and set classifiers and parameter lists
    classifiers = list(classifiers_params.keys())
    params_list = list(classifiers_params.values())
    
    # Iterate over each classifier and run GridSearchCV using the given params
    for i in range(len(classifiers)):
        print('\n\nTuning classifier...')
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
                            cv = cross_val)
        
        # Check and resize data shape, then fit on grid
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
        features_scores = ['%.2f' % elem for elem in grid.best_estimator_.named_steps['selector'].scores_]
        features_selected_scores = [x for x, y in zip(features_scores, features_selected_bool) if y]

        print('\nThe features used are:')
        for i in range(len(features_selected_list)):
            print(str(features_selected_list[i]) + ' ' + str(features_selected_scores[i]))
        
        # Get best estimator for classifier
        print('\nBest estimator = \n' + str(grid.best_estimator_))
        best_estimators.update({f1_score : grid.best_estimator_})   
        
        # Run test_classifer
        print('\n\nRunning Tester...\n' )#+ str(type(classifiers[i])))
        test_classifier(grid.best_estimator_, my_dataset, features_list)
    
    best_score = max(list(best_estimators.keys()))
    
    return best_estimators[best_score]

    
def dumpClf(clf, my_dataset, features_list):
    '''
    clf: Best classifier given the scoring metric (a pipeline object)
    
    my_dataset: Dict of people in Eron, with values of dictonaries (a dict)
        
    features_list: list of labels for features of Enron dataset (a list) 
    '''
    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.
    dump_classifier_and_data(clf, my_dataset, features_list)


def main():
    # Get, create, explore, and store Enron dataset
    feature_names = getFeatureList()
    dataset = getDataDict()
    evaluateDataset(dataset)
    dataset = removeOutliers(dataset)
    #dataset = createFeatures(dataset)
    
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
    
    # Select final classifiers to tune
    final_classifier = getClassiferTunes([GaussianNB()], classifiers_params_list)
    
    ###### *** Uncomment line below to tune all classifiers *** #####
    ###### *** WARNING: RUNTIME MAY BE EXTREMELY LONG*** ######
    #final_classifier = getClassiferTunes(classifiers, classifiers_params_list)
                        
    # Tune given Classifiers
    classifier = tuneClassifier(final_classifier, cv, dataset, feature_names,
                   features_train, labels_train, features_test, labels_test)
    
    # Dump classifier, dataset, and features to tester.py
    dumpClf(classifier, dataset, feature_names)
    
    
if __name__ == '__main__':
    main()
