import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
##%matplotlib inline
adv = pd.read_csv('C:\\Users\\nidhchoudhary\\OneDrive - Deloitte (O365D)\\UDEMY\\Py_DS_ML_Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\13-Logistic-Regression\\advertising.csv')
adv.head()
sns.heatmap(adv.isnull())
##To check how many people clicked on add
sns.countplot(x= 'Clicked on Ad',data = adv)
##Time spent on Internet
sns.distplot(adv['Daily Time Spent on Site'],bins = 50)
adv.drop(['Ad Topic Line','City','Country','Timestamp'],axis = 1,inplace = True)
x = adv.iloc[0:,0:5]
x.head()
y = adv.iloc[0:,5:6]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 0.3)
x_train.head()
x_test.head()
y_train.head()
y_test.head()
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
model.coef_
model.intercept_
y_predict = model.predict(x_test)
conf_mat
from sklearn.metrics import mean_absolute_error
mean_absolute = mean_absolute_error(y_test,y_predict)
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test,y_predict)
print(rmse)
from sklearn.metrics import f1_score
f1_score(y_test,y_predict)