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

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### create linear classifier
#clf = SVC(kernel = 'linear')

### create rbf classifier
clf = SVC(kernel = 'rbf')
                            
### start timer
t0 = time()

### cut down on processing time by reducing features to 1% of original
features_train_1_percent = features_train[:len(features_train)/100] 
labels_train_1_percent = labels_train[:len(labels_train)/100] 


### fit the classifier on the training features and labels
clf.fit(features_train_1_percent, labels_train_1_percent)

### stop timer
print('Training Time: ', round(time() - t0, 3), "s")


### start timer
t1 = time()

### use the trained classifier to predict labels for the test features
pred = clf.predict(features_test)

### stop timer
print('Prediction Time: ', round(time() - t1, 3), "s")

### calculate and return the accuracy on the test data
accuracy = clf.score(features_test, labels_test)


print(accuracy)



