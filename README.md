ud120-projects
==============

Starter project code for students taking Udacity ud120

# Introduction
## Summary
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

## Resources Needed
You should have python and sklearn running on your computer, as well as the starter code (both python scripts and the Enron dataset) that you downloaded as part of the first mini-project in the Intro to Machine Learning course. You can get the starter code on git: git clone https://github.com/udacity/ud120-projects.git

The starter code can be found in the final_project directory of the codebase that you downloaded for use with the mini-projects. Some relevant files: 

poi_id.py : Starter code for the POI identifier, you will write your analysis here. You will also submit a version of this file for your evaluator to verify your algorithm and results. 

final_project_dataset.pkl : The dataset for the project, more details below. 

tester.py : When you turn in your analysis for evaluation by Udacity, you will submit the algorithm, dataset and list of features that you use (these are created automatically in poi_id.py). The evaluator will then use this code to test your result, to make sure we see performance that’s similar to what you report. You don’t need to do anything with this code, but we provide it for transparency and for your reference. 

emails_by_address : this directory contains many text files, each of which contains all the messages to or from a particular email address. It is for your reference, if you want to create more advanced features based on the details of the emails dataset. You do not need to process the e-mail corpus in order to complete the project.


# Results
## Enron Submission Free-Response Questions

### Dataset Summary
* Enron dataset contains 146 people.
* Enron dataset contains 18 People of Interest.
* Enron dataset contains 128 Non-People of Interest.

1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]
 * The goal of this project is to create a tuned machine learning classifier in order to determine whether or not an employee of Enron is a Person of Interest based on the features of each person given within the dataset. This dataset is one of a collection of one of, if not the, largest real financial sandals to ever occur within a single company. Enron committed systematic financial fraud, and when discovered numerous people went to jail, even more were People of Interest in the investigation. This machine learning classifier attempts to determine whether or not an employee of Enron is a Person of Interest based on the features of each person given within the dataset. In this dataset there was one outlier that was removed before the machine learning classifier was fit on the data, 'TOTAL'. 'TOTAL' was a bad sample input due to the way in which the original spreadsheet of the dataset was structured. 'TOTAL' included the summation of each feature for all of the samples in the dataset. 'TOTAL' was found by viewing the dataset through visual exploration, and then further in depth through the provided Enron dataset documentation in final_project/enron61702insiderpay.pdf.

2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importance of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]
 * SelectKBest was used for the feature selection for the classifier pipeline. As part of the assignment, 'email_address' was removed as a feature due to this feature being used for the email data from the dataset, which was not used in this classifier. A new feature was created, but was not used in the classifier due to it not being selected by SelectKBest in the pipeline. 'poi_email_ratio', which was the ratio of total to and from 'poi_emails' divided by the total to and from emails for each person. This feature had no effect on the untuned Naive Bayes in the metrics of accuracy, percision, recall, f1 score, or AUROC numerical metrics. This is most likely due to the new feature being dependent on 3 other features in the dataset being used for classification, 'poi_emails', 'to_emails', and 'from_emails'. Feature scaling was used within the classifier pipeline, MinMaxScalar was the selected feature scalar function. Given the wide range of values for features from feature to feature, MinMaxScalar was selected. From the SelectKBest, these features and scores were found:
 - salary 15.81
 - total_payments 8.96
 - exercised_stock_options 9.96
 - bonus 30.65
 - restricted_stock 8.05
 - shared_receipt_with_poi 10.67
 - total_stock_value 10.81
 - loan_advances 7.04
 - deferred_income 8.49
 - long_term_incentive 7.53
 - Total of 10 features used, k = 10


3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
 * Naive Bayes was the final algorithm used for the classifier pipeline. The factors that lead to choosing Naive Bayes, was the overall performance during iterations of the pipeline. Naive Bayes consistently scored, on average, higher than the other classifiers. The runtime of Naive Bayes within the pipeline is much lower than the other classifiers due to lack of parameter dimensional. The following are the results from untuned classifiers, with no feature reduction, feature scaling, parameter tuning, or PCA. These results are from the evaluateClf() method, and utilizes the sklearn metrics module.
 * **KNeighbors**
    * Accuracy = 0.886363636364
    * Percision = 0.0
    * Recall = 0.0
    * F1 Score = 0.0
    * ROC Curve AUC = 0.5
 * **SVM**
    * Accuracy = 0.886363636364
    * Percision = 0.0
    * Recall = 0.0
    * F1 Score = 0.0
    * ROC Curve AUC = 0.5
 * **Decision Tree**
    * Accuracy = 0.863636363636
    * Percision = 0.333333333333
    * Recall = 0.2
    * F1 Score = 0.25
    * ROC Curve AUC = 0.574358974359
 * **Random Forest**
    * Accuracy = 0.863636363636
    * Percision = 0.333333333333
    * Recall = 0.2
    * F1 Score = 0.25
    * ROC Curve AUC = 0.574358974359
 * **AdaBoost**
    * Accuracy =  0.840909090909
    * Percision = 0.25
    * Recall = 0.2
    * F1 Score = 0.222222222222
    * ROC Curve AUC = 0.561538461538
 * **Naive Bayes**
    * Accuracy = 0.886363636364
    * Percision = 0.15
    * Recall = 0.4
    * F1 Score = 0.444444444444
    * ROC Curve AUC = 0.674358974359
4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]
 * By passing a range of values for each parameter, allows for GridSearchCV to exhaustively search all combinations of parameter values until the best estimator is found for the pipeline. Naive Bayes was used as the final classifier for the pipeline. Other algorithms were tuned in iteration testing by using GridSearchCV to create a pipeline of various parameter options given each classifier type. While PCA, SelectKBest, andMinMaxScalar, all kept the same range of optional values for their respective parameters for each classifier type. While tuning a classifier, or a pipeline containing a classifier, using GridSearchCV. For each parameter that is to be tuned for a given classifier, a list of values for each parameter must be given. Increasing the options of values within each parameter, increases the dimensional of GridSearchCV, and can greatly impact the runtime of GridSearchCV. This was not as much of an issue with Naive Bayes, given that this classifier did not have any list of values for each parameter passed into GridSearchCV.

5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]
 * A validation, or cross validation, is the process of randomly splitting the given training dataset so that each fit of the machine learning classifier can be validated against a test subset of the dataset. By properly validating the machine learning classifier, overfitting of the dataset can be better avoided. The cross validation used in this project was StratifiedShuffleSplit().

6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
 * The evaluation metrics found for the final classifier used within the pipeline on the Enron dataset on the local Enron dataset testing subset were:
    * Accuracy = 0.863636363636
    * Percision = 0.4
    * Recall = 0.4
    * F1 Score = 0.4
    * ROC Curve AUC = 0.661538461538
 * The recall for this classifier was, 0.4. Which tells us the proportion of people in the dataset that were actually People of Interest and were predicted by the classifier as being a Person of Interest. That is, 40.0% of true positives People of Interest in the dataset of true People of Interest.

 * The precision for this classifier was, 0.4. The precision metric showed that out of the people in the testing subset, out of all the people classified as People of Interest, only 40.0% were classified correctly as People of Interest.

Final Metrics Given by tester.py:

* Accuracy: 0.83960       
* Precision: 0.37682      
* Recall: 0.31050 
* F1: 0.34046     
* F2: 0.32183
* Total predictions: 15000        
* True positives:  621    
* False positives: 1027   
* False negatives: 1379   
* True negatives: 11973


# References
* http://scikit-learn.org/stable/index.html
* https://discussions.udacity.com/t/adding-pipeline-for-pca-onto-gridsearchcv/226682
* https://discussions.udacity.com/t/how-to-find-out-the-features-selected-by-selectkbest/45118/4
