# Credit Risk Analysis Report
## Overview of the Analysis
* Purpose of the analysis
  * To identify the creditworthiness of borrowers by building a logistic regression model using the historical lending activity dataset from a peer-to-peer lending services company.

* Financial information of the data
  * The historical lending activity data provides information such as loan size, interest rate, borrower’s income, debt to income, number of accounts, derogatory marks, and total loans. The credit worthiness of borrowers is to be predicted using the label loan_status label.
  * A look at the dataset using value_counts reveals that the dataset is not well-balanced data with data labeled 0 (healthy loan) amounting to 96.77% and data labeled 1 (high-risk loan) amounting to just 3.22%.

* Stages of the machine learning process
  * The lending history data read from the file is preprocessed by separating the features from the target variable 'loan_status'. The data is then split into training and test sets. An initial check of the target variable shows imbalanced classes, with more examples of 0 (healthy loan) amounting to 96.77% over 1 (high-risk loan) amounting to just 3.22%.
  * A logistic regression model is trained on the training data and makes predictions on the held-out test set. The model's performance is evaluated using accuracy, confusion matrix, and a classification report, which provides an overall performance score, per-class performance, and granular metrics like precision and recall.
  * To handle the class imbalance, the training data is resampled using RandomOverSampler to balance the target variable distribution. A new logistic regression model is trained on the resampled data and evaluated again.

* Resampling using RandomOverSampler 
  * The imbalance in the original training data led to disparate model performance between the healthy and high-risk loan classes. To mitigate this class imbalance, RandomOverSampler was employed to resample the training data.
  
  * By oversampling the minority high-risk loan class, RandomOverSampler produced a balanced training set with equal representation of both classes. Retraining the model on this resampled data improved performance in the high-risk class.
  
  * The new model increased precision to 0.84, recall to 0.99, and F1 to 0.91 for the high-risk loans. The gap in performance metrics between the classes has now been reduced substantially.
  
  * Using RandomOverSampler to rebalance the training data distribution addressed the original class imbalance. The retrained model on the resampled data enhances identification of the previously underperforming minority class. This demonstrates how resampling can improve model performance on skewed datasets.

## Results

* Machine Learning Model 1:Logistic regression with original data
*             precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

     accuracy                          0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384


From the classification report it is seen that the logistic regression model performs very well for class 0 (healthy loan). The precision is 1.00, recall is 0.99 and f1-score is 1.00,
The model performs moderately well to classify the class 1 (high-risk loan) labels, with comparatively lower precision(0.85), recall (0.91) and f1-score(0.88).



* Machine Learning Model 2:Logistic regression with resampled data
*                 precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.99      0.91       619

    accuracy                           0.99     19384
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384


The logic regression model with the resampled training data, performed slightly better than the logic regression model with the original data in terms of recall and f1-score for class 1 (high-risk loan) label.
The performance of this model is same as that for model with original data for 0 (healthy loan).

## Summary
The new model increased precision to 0.84, recall to 0.99, and F1 to 0.91 for the high-risk loans. The gap in performance metrics between the classes has now reduced substantially.
The performance of this model is same as that for model with original data for 0 (healthy loan).

