# Enron Submission Free-Response Questions

A critical part of machine learning is making sense of your analysis process and communicating it to others. The questions below will help us understand your decision-making process and allow us to give feedback on your project. Please answer each question; your answers should be about 1-2 paragraphs per question. If you find yourself writing much more than that, take a step back and see if you can simplify your response!

When your evaluator looks at your responses, he or she will use a specific list of rubric items to assess your answers. Here is the link to that rubric: Link to the rubric Each question has one or more specific rubric items associated with it, so before you submit an answer, take a look at that part of the rubric. If your response does not meet expectations for all rubric points, you will be asked to revise and resubmit your project. Make sure that your responses are detailed enough that the evaluator will be able to understand the steps you took and your thought processes as you went through the data analysis.

Once you’ve submitted your responses, your coach will take a look and may ask a few more focused follow-up questions on one or more of your answers.  

## We can’t wait to see what you’ve put together for this project!

1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]
 * The goal of this project is to create a tuned machine learning classifier in order to determine whether or not an employee of Enron is a Person of Interest based on the features of each person given within the dataset. This dataset is one of a collection of one of, if not the, largest real financial sandals to ever occur within a single company. Enron committed systematic financial fraud, and when discovered numerous people went to jail, even more were People of Interest in the investigation. This machine learning classifier attempts to determine whether or not an employee of Enron is a Person of Interest based on the features of each person given within the dataset. In this dataset there was one outlier that was removed before the machine learning classifier was fit on the data, 'TOTAL'. 'TOTAL' was a bad sample input due to the way in which the original spreadsheet of the dataset was structured. 'TOTAL' included the summation of each feature for all of the samples in the dataset.

2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

 * KNeighbors 
   Accuracy = 0.863636363636
   Percision = 0.0
   Recall = 0.0
   F1 Score = 0.0

 * SVM
   Accuracy = 0.863636363636
   Percision = 0.0
   Recall = 0.0
   F1 Score = 0.0

 * Gaussian Process 
   Accuracy = 0.863636363636
   Percision = 0.0
   Recall = 0.0
   F1 Score = 0.0

 * Decision Tree
   Accuracy = 0.704545454545
   Percision = 0.111111111111
   Recall = 0.166666666667
   F1 Score = 0.133333333333

 * Random Forest
   Accuracy = 0.863636363636
   Percision = 0.5
   Recall = 0.166666666667
   F1 Score = 0.25

 * Neural Network
   Accuracy = 0.795454545455
   Percision = 0.0
   Recall = 0.0
   F1 Score = 0.0

 * AdaBoost
   Accuracy = 0.909090909091
   Percision = 1.0
   Recall = 0.333333333333
   F1 Score = 0.5

 * Naive Bayes
   Accuracy = 0.295454545455
   Percision = 0.162162162162
   Recall = 1.0
   F1 Score = 0.279069767442

 * Quadratic Discriminant Analysis
   Accuracy = 0.840909090909
   Percision = 0.0
   Recall = 0.0
   F1 Score = 0.0



4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]
 * A validation, or cross validation, is the process of randomly splitting the given training dataset so that each fit of the machine learning classifier can be validated against a test subset of the dataset.

6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]