#!/usr/bin/python

"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from time import time
from sklearn.cross_validation import train_test_split
from sklearn import metrics

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# split training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!  
### create classifier
clf = tree.DecisionTreeClassifier()

### start timer
t0 = time()

### fit the classifier on the training features and labels
clf = clf.fit(features_train, labels_train)

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


print('Accuracy = '
      + str(accuracy))


# print number of POI predicted by DT
print('Number of predicted POI = '
      + str(sum(pred)))

print('Number of poeple in test set = '
      + str(len(labels_test)))

# print out the number of true positives
true_positive_count = 0

for i in range(len(pred)):
    if pred[i]:
        if labels_test[i]:
            true_positive_count += 1
            
print('Number of True Positives = '
      + str(true_positive_count))

print('Percision = '
      + str(metrics.precision_score(labels_test, pred)))

print('Recall = '
      + str(metrics.recall_score(labels_test, pred)))

