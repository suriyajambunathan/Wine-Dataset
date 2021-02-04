#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: suriyaprakashjambunathan
"""

# TASK 4
        
#Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

#Importing PCA and the Classifiers
from sklearn.decomposition import PCA               # PCA
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.linear_model import LinearRegression   # Linear Regression
from sklearn.svm import SVC                         # SVM 
from sklearn.naive_bayes import GaussianNB          # Naive Bayesian

# defining the classifier function
def clf (X_train,y_train,X_test,y_test,clfr):
    
    #assigning the classifier based on the type of classifier
    if clfr == "logreg":
        classifier = LogisticRegression(random_state = 0)
    elif clfr == "linreg":
        classifier = LinearRegression()
    elif clfr == "svm":
        classifier = SVC(kernel = 'linear', random_state = 0)
    elif clfr == "nb":
        classifier = GaussianNB()
        
    #fitting the classifier model on the dataset
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    qual_pred = classifier.predict(X_test)
    
    if clfr == "linreg":
        for i in range(len(qual_pred)):
            qual_pred[i] = int(round(qual_pred[i]))
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, qual_pred)

    # Calculating the measurement parameters
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    acc = (TP+TN)/(TP+FP+FN+TN)
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    F_mes = 2*((TP/(TP+FN)) * (TP/(TP+FP))) / ((TP/(TP+FN)) + (TP/(TP+FP)))
    spec = TN/(TN+FP)
    sens = TP/(TP+FN)
    
    return(acc,pre,rec,F_mes,spec,sens)
 
#Importing the datasets
column_names = ["fixed acidity","volatile acidity","citric acid","residual sugar",
                "chlorides","free sulfur dioxide","total sulfur dioxide","density","pH",
                "sulphates","alcohol","quality"]
index_val = ["Logistic Regression","Linear Regression","SVM","Naïve Bayesian"]
column_val = ["Accuracy","Precision","Recall","F-Measure","Specificity","Sensitivity"]

#train set
train_arr = np.genfromtxt('wineQualityRed_train.csv', delimiter=";",skip_header = 1)
train_set = pd.DataFrame(data = train_arr,   
                  columns = column_names) 

#test set
test_arr = np.genfromtxt('wineQualityRed_test.csv', delimiter=";",skip_header = 1)
test_set = pd.DataFrame(data = test_arr,   
                  columns = column_names) 

#Quality determination based on quality value
qual_train = []
qual_test = []
for i in (train_set['quality']):
    if i >= 7.0 :
        qual_train.append(1)
    else:
        qual_train.append(0)
for i in (test_set['quality']):
    if i >= 7.0 :
        qual_test.append(1)
    else:
        qual_test.append(0)
 
#Removing the quality attribute for applying PCA
X_train = pd.DataFrame(data = train_set.iloc[:, 0:-1].values) 
X_test = pd.DataFrame(data = test_set.iloc[:, 0:-1].values)    
 
# Applying PCA
pca = PCA(n_components = 11)
X_train = pd.DataFrame(data = pca.fit_transform(X_train),
                       columns = column_names[:-1])
X_test = pd.DataFrame(data = pca.transform(X_test),
                      columns = column_names[:-1])
explained_variance = pca.explained_variance_ratio_

#Concatenating the quality attribute after applying PCA on other elements
X_train = pd.concat([X_train, train_set["quality"]], axis=1)
X_test = pd.concat([X_test, test_set["quality"]], axis=1)



# 7 attributes

redwine_7_training = X_train.iloc[:, 0:7].values    
redwine_7_testing  = X_test.iloc[:, 0:7].values 

#Logistic Regression
acc_logreg,prec_logreg,rec_logreg,F_mes_logreg,spec_logreg,sens_logreg\
= clf(redwine_7_training,qual_train,redwine_7_testing,qual_test,"logreg")

#Linear Regression as a Classifier
acc_linreg,prec_linreg,rec_linreg,F_mes_linreg,spec_linreg,sens_linreg\
= clf(redwine_7_training,qual_train,redwine_7_testing,qual_test,"linreg")
  
#SVM
acc_svm,prec_svm,rec_svm,F_mes_svm,spec_svm,sens_svm\
= clf(redwine_7_training,qual_train,redwine_7_testing,qual_test,"svm")  

#Naïve Bayesian     
acc_nb,prec_nb,rec_nb,F_mes_nb,spec_nb,sens_nb\
= clf(redwine_7_training,qual_train,redwine_7_testing,qual_test,"nb")   

#Displaying the parameters table
task_4 = [[acc_logreg,prec_logreg,rec_logreg,F_mes_logreg,spec_logreg,sens_logreg],
          [acc_linreg,prec_linreg,rec_linreg,F_mes_linreg,spec_linreg,sens_linreg],
          [acc_svm,prec_svm,rec_svm,F_mes_svm,spec_svm,sens_svm],
          [acc_nb,prec_nb,rec_nb,F_mes_nb,spec_nb,sens_nb]]

attr_7_df = pd.DataFrame(data = task_4,
                        index  = index_val,
                        columns = column_val
                        ) 

print(attr_7_df)       



# 4 attributes

redwine_4_training = X_train.iloc[:, 0:4].values    
redwine_4_testing  = X_test.iloc[:, 0:4].values 

#Logistic Regression
acc_logreg,prec_logreg,rec_logreg,F_mes_logreg,spec_logreg,sens_logreg\
= clf(redwine_4_training,qual_train,redwine_4_testing,qual_test,"logreg")

#Linear Regression as a Classifier
acc_linreg,prec_linreg,rec_linreg,F_mes_linreg,spec_linreg,sens_linreg\
= clf(redwine_4_training,qual_train,redwine_4_testing,qual_test,"linreg")
  
#SVM
acc_svm,prec_svm,rec_svm,F_mes_svm,spec_svm,sens_svm\
= clf(redwine_4_training,qual_train,redwine_4_testing,qual_test,"svm")  

#Naïve Bayesian     
acc_nb,prec_nb,rec_nb,F_mes_nb,spec_nb,sens_nb\
= clf(redwine_4_training,qual_train,redwine_4_testing,qual_test,"nb")   

#Displaying the parameters table
task_4 = [[acc_logreg,prec_logreg,rec_logreg,F_mes_logreg,spec_logreg,sens_logreg],
          [acc_linreg,prec_linreg,rec_linreg,F_mes_linreg,spec_linreg,sens_linreg],
          [acc_svm,prec_svm,rec_svm,F_mes_svm,spec_svm,sens_svm],
          [acc_nb,prec_nb,rec_nb,F_mes_nb,spec_nb,sens_nb]]

attr_4_df = pd.DataFrame(data = task_4,
                        index  = index_val,
                        columns = column_val
                        ) 

print(attr_4_df)       