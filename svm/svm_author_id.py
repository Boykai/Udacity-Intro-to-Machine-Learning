#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from collections import Counter

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### create linear classifier
#clf = SVC(kernel = 'linear')

c = 10000

### create rbf classifier
clf = SVC(kernel = 'rbf', C = c)

print('\nC = ' + str(c))
                            
### start timer
t0 = time()

### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)

### stop timer
print('Training Time: ', round(time() - t0, 3), "s")


### start timer
t1 = time()

### use the trained classifier to predict labels for the test features
pred = clf.predict(features_test)

### stop timer
print('Prediction Time: ', round(time() - t1, 3), "s")

### print the requested prediction values
print('Prediction 10 = ' + str(pred[10]))
print('Prediction 26 = ' + str(pred[26]))
print('Prediction 50 = ' + str(pred[50]))


### calculate and return the accuracy on the test data
accuracy = clf.score(features_test, labels_test)

print(accuracy)

### get number of 0 and 1 counts for prediction results
counts = Counter(pred)

print(counts)


