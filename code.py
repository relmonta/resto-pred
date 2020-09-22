# data analysis and wrangling
import pandas as pd
import numpy as np
import random
import datetime

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# indexer le City Group
train_df['CityGroup']=train_df['City Group'].apply(lambda x: 1 if x=="Big Cities" else 0)
test_df['CityGroup']=test_df['City Group'].apply(lambda x: 1 if x=="Big Cities" else 0)

# calculer l'age du resto en jours
def datediff(y,d,m):
    x = datetime.date(y, m, d)
    y = datetime.date.today()
    return (y-x).days

train_df['OpenDate']=train_df['Open Date']
test_df['OpenDate']=test_df['Open Date']
train_df['OpenDate']=train_df['OpenDate'].apply(
    lambda x: datediff(int(x.split("/")[2]),int(x.split("/")[1]),int(x.split("/")[0])))
test_df['OpenDate']=test_df['OpenDate'].apply(
    lambda x: datediff(int(x.split("/")[2]),int(x.split("/")[1]),int(x.split("/")[0])))

print(train_df.City.value_counts())
print(test_df.City.value_counts())

