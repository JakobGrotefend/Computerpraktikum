# -*- coding: utf-8 -*-
"""
Explorative analysis of the bike sharing data

@author: Jakob Grotefend
"""

import pandas as pd
## import bike sharing data
data = pd.read_csv("hour.csv")

# histogramm full
data.drop(["casual", "registered"], axis=1).hist(bins=50,figsize = (20,15))

# histogramm non-categorical
data[["atemp", "cnt", "hum", "temp", "windspeed"]].hist(bins=50, figsize = (20,15))


X = data.drop(["casual","registered","cnt"], axis=1) # independent variables
y_cnt = data["cnt"].astype(float) # dependent variable (the final model will predict the total number of bike usages "cnt")


## exploratory data analysis on full data set
print("Some data samples:\n",data.head()) # overview of the dataset
print("Information about the variables:")
data.info() # further information about the individual variables. No missing values and no strings
attrib_selection_hist = ["atemp","cnt","hr","hum","temp","weathersit","windspeed"]
data.hist(bins=50, figsize = (20,15))
data[attrib_selection_hist][data["hr"]==7].hist(bins=50, figsize = (20,20)) # plot histograms  of each variable

# split data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_cnt_train, y_cnt_test = train_test_split(X, y_cnt, test_size=0.2, random_state=42)

# further exploratory data analysis
data_explore = pd.concat([X_train, y_cnt_train], axis = 1, sort=False) # data matrix containing both independent and dependent variables

corr_matrix = data_explore[data_explore["hr"]==7].corr() # correlation matrix
print("The correlation of the independent variables with the dependent variable: \n", corr_matrix["cnt"]) # correlation matrix for the dependent variable "cnt"
# scatter plots of a selection of variables
from pandas.plotting import scatter_matrix
attrib_selection_scatter = ["instant", "season","temp","atemp", "hum","windspeed","cnt"]
scatter_matrix(data_explore[attrib_selection_scatter][data_explore["hr"]==7],figsize=(12,8)) # scatter plots with holding hour constant
scatter_matrix(data_explore[attrib_selection_scatter][data_explore["hr"]==7][data_explore["workingday"]==1][["instant","cnt"]],figsize=(12,8)) # scatter plots with holding hour and workingday constant

data_explore[attrib_selection_scatter][data_explore["hr"]==7][data_explore["workingday"]==1][["instant","cnt"]].plot.scatter(x="instant", y="cnt") # scatter plot of instant vs cnt



