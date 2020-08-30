import pandas as pd
import numpy as np 
import matplotlib.pyplot as py 
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

bank = pd.read_csv('C:/Users/nidhchoudhary/Desktop/Assignment/LOGISTIC_REGRESSION/bank_data.csv')

bank.head(10)
#print(bank.head(10))



x = bank.loc[:, bank.columns != 'y']

print(x.head())

Y = bank.y


print(Y.head())

function = LogisticRegression()

Y_model = (function.fit(x,Y))
Y_Pred =  function.predict(x)

print(Y_Pred)

bank["Y_PREDICTED_VALUES"] = Y_Pred
print(bank.head(10))

from sklearn.metrics import confusion_matrix
confusion_metric = metrics.confusion_matrix(Y,Y_Pred)
print(confusion_metric)

print(len(bank.y))


accuracy = sum(bank.y==bank.Y_PREDICTED_VALUES)/len(bank.y)
print(accuracy)

