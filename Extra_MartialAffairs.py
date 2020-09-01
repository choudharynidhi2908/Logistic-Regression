import pandas as pd
import numpy as np 
import matplotlib.pyplot as py
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

affairs = pd.read_csv('C:/Users/nidhchoudhary/Desktop/Assignment/LOGISTIC_REGRESSION/affairs.csv')

#print(affairs.head())

affairs = affairs.drop(affairs.columns[0],axis=1)


print(affairs.head())

y = affairs.naffairs

print(y)

x= affairs.iloc[:,1:]

print(x)
function = LogisticRegression()

y_model1 = (function.fit(x,y))
Y_Predict = function.predict(x)

print(Y_Predict)

affairs['Y_PRED'] = Y_Predict

print(affairs.head())

from sklearn.metrics import confusion_matrix
confusion_metrics = metrics.confusion_matrix(y,Y_Predict)

accuracy = sum(y==Y_Predict)/len(affairs.naffairs)

print(accuracy)
