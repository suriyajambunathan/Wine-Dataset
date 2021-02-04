#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: suriyaprakashjambunathan
"""

# TASK 2
  
#Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Importing PCA and the Classifiers
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
    
    return(cm)
    
    
    
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
        
train_set = pd.DataFrame(data = train_set.iloc[:, 0:-1].values) 
test_set = pd.DataFrame(data = test_set.iloc[:, 0:-1].values)

#Logistic Regression
logreg_cm = clf(train_set,qual_train,test_set,qual_test,"logreg")

#Linear Regression as a Classifier
linreg_cm = clf(train_set,qual_train,test_set,qual_test,"linreg")
  
#SVM
svm_cm = clf(train_set,qual_train,test_set,qual_test,"svm")  

#Naïve Bayesian     
nb_cm = clf(train_set,qual_train,test_set,qual_test,"nb")   

print(" ")
print("Logistic Regression Confusion Matrix")
print(" ")
print(logreg_cm)
print(" ")
print("Linear Regression Confusion Matrix")
print(" ")
print(linreg_cm)
print(" ")
print("SVM Confusion Matrix")
print(" ")
print(svm_cm)
print(" ")
print("Naïve Bayesian  Confusion Matrix")
print(" ")
print(nb_cm)



                        
