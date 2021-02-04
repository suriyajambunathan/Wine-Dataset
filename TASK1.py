#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: suriyaprakashjambunathan
"""

# TASK 1

'''def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])'''

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''import subprocess
import sys
install('seaborn')
import seaborn as sns'''

#Importing the datasets

column_names = ["fixed acidity","volatile acidity","citric acid","residual sugar",
                "chlorides","free sulfur dioxide","total sulfur dioxide","density","pH",
                "sulphates","alcohol","quality"]

#train set
train_arr = np.genfromtxt('wineQualityRed_train.csv', delimiter=";",skip_header = 1)
train_set = pd.DataFrame(data = train_arr,   
                  columns = column_names) 

#test set
test_arr = np.genfromtxt('wineQualityRed_test.csv', delimiter=";",skip_header = 1)
test_set = pd.DataFrame(data = test_arr,   
                  columns = column_names) 

#Splitting the datasets into independent and dependent variables
train_set_vararr = train_set.iloc[:, :-1].values
train_set_var =  pd.DataFrame(data = train_set_vararr,   
                  columns = column_names[:-1]) 

train_set_qualarr = train_set.iloc[:, -1].values
train_set_qual =  pd.DataFrame(data = train_set_qualarr,   
                  columns = [column_names[-1]]) 

test_set_vararr = test_set.iloc[:, :-1].values
test_set_var =  pd.DataFrame(data = test_set_vararr,   
                  columns = column_names[:-1]) 

test_set_qualarr = test_set.iloc[:, -1].values
test_set_qual =  pd.DataFrame(data = test_set_qualarr,   
                  columns = [column_names[-1]]) 

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_set_var, train_set_qual)

coeff = regressor.coef_


# Predicting the Test set results
qual_pred = regressor.predict(test_set_var)


# Visualising the Training set results
'''f, ax = plt.subplots(figsize=(10, 8))
corr = train_set.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.title("Linear Regression", fontsize =40)'''

#plotting the graph
X_train = pd.DataFrame(data = train_set["volatile acidity"])
y_train = pd.DataFrame(data = train_set["quality"])

X_test = pd.DataFrame(data = test_set["volatile acidity"])
y_test = pd.DataFrame(data = test_set["quality"])


regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Volatile Acidity vs Quality (Training set)')
plt.xlabel("Volatile Acidity")
plt.ylabel('Quality')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Volatile Acidity vs Quality (Test set)')
plt.xlabel("Volatile Acidity")
plt.ylabel('Quality')
plt.show()

#Calculating Sum of Square Error for Test Dataset
sse = 0
for i in range(len(qual_pred)):
    sse = (qual_pred[i] - list(test_set_qualarr)[i])**2
mse = sse/len(qual_pred)

print(' ')
print("The Sum of Square Errors for Test Dataset: ")
print(sse[0])

print(' ')
print('On dividing with the total number of elements to find the average')
print(' ')

print("Mean Square Error for Test Dataset: ")
print(mse[0])

print(' ')
print("The Coefficients of Linear Regression are: ")
for i in coeff[0]:
    print(i)
