# data analysis and wrangling
import pandas as pd
import numpy as np
import random
import datetime

# visualization
#import seaborn as sns
#import matplotlib.pyplot as plt
# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.svm import SVC, LinearSVC

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# calculer l'age du resto en jours
def datediff(y,d,m):
    x = datetime.date(y, m, d)
    y = datetime.date.today()
    return (y-x).days

Type = pd.get_dummies(train_df['Type'])
Type['MB']=0
train_df=train_df.join(Type)
Type = pd.get_dummies(test_df['Type'])
test_df=test_df.join(Type)

citygroup = pd.get_dummies(train_df['City Group'])
train_df=train_df.join(citygroup)
citygroup = pd.get_dummies(test_df['City Group'])
test_df=test_df.join(citygroup)

train_df['AgeDays']=train_df['Open Date']
test_df['AgeDays']=test_df['Open Date']
train_df['AgeDays']=train_df['AgeDays'].apply(
    lambda x: datediff(int(x.split("/")[2]),int(x.split("/")[1]),int(x.split("/")[0])))
test_df['AgeDays']=test_df['AgeDays'].apply(
    lambda x: datediff(int(x.split("/")[2]),int(x.split("/")[1]),int(x.split("/")[0])))

#print(train_df.City.value_counts())
#print(test_df.City.value_counts())

X = train_df.copy()
X_test = test_df.copy()
X.drop(['Open Date', 'City','City Group','Type','revenue'], axis = 1, inplace = True)
X_test.drop(['Open Date', 'City','City Group','Type'], axis = 1, inplace = True)

y = train_df['revenue'].copy()

reg = RandomForestRegressor().fit(X, y)
print(reg.score(X, y))
predict_test=reg.predict(X_test)
pd.DataFrame(predict_test).to_csv("submission.csv", header="Prediction", index="Id")