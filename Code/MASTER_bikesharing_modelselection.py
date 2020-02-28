# -*- coding: utf-8 -*-
"""
Model selection for the bike sharing dataset.

@author: Jakob Grotefend
"""

import numpy as np
import pandas as pd

## import bike sharing data
data = pd.read_csv("hour.csv")
X = data.drop(["casual","registered","cnt"], axis=1) # independent variables
y_cnt = data["cnt"].astype(float) # dependent variable (the final model will predict the total number of bike usages "cnt")


## data preprocessing
# split data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_cnt_train, y_cnt_test = train_test_split(X, y_cnt, test_size=0.2, random_state=42)

# create a custom transformer for data preperation
# import base classes for custom transformers
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

# transformer that drops some redundant variables
class DelRedundantVal(BaseEstimator,TransformerMixin):
    def __init__(self, feature_drop = ["dteday","season"]):
        self._feature_drop = feature_drop
    def fit(self, X, y=None):
        return self
    def transform(self, X,y=None):
        return X.drop(self._feature_drop,axis=1)

# transformer to add additional powers of hr and mnth
class AttributeAdder(BaseEstimator,TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X,y=None):
        return self
    def transform(self, X,y=None):
        X["hr2"] = X["hr"].astype(float)*X["hr"].astype(float)
        X["hr3"] = X["hr2"].astype(float)*X["hr"].astype(float)
        X["mnth2"] = X["mnth"].astype(float)*X["mnth"].astype(float)
        X["mnth3"] = X["mnth2"].astype(float)*X["mnth"].astype(float)
        X["temp2"] = X["temp"].astype(float)*X["temp"].astype(float)
        X["temp3"] = X["temp2"].astype(float)*X["temp"].astype(float)
        return X

# transformer to cast data values as float64
class FloatCaster(BaseEstimator,TransformerMixin):
    def __init__(self):
        return None
    def fit(self,X,y=None):
        return self
    def transform(self, X,y=None):
        return X.astype(float).values

# create a pipeline to transform the data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

dataprep_pipeline = Pipeline([
        ("redundand_val", DelRedundantVal()),
        ("attribute_adder", AttributeAdder()),
        ("float_caster",FloatCaster()),
        ("standard_scaler", StandardScaler())
    ])

X_train_prepared = dataprep_pipeline.fit_transform(X_train) # preprocessed training set


## testing different models
from sklearn.model_selection import cross_val_score # import cross-validation tool for model evaluation
eval_list = [] # list to store the performance of the tested modeles

# function for printing the performance evaluations
def print_eval(evallist):
    for model in evallist:
        print(model[0]," : ",model[1]," , ", model[2])

# function for printing the execution time
def print_time(evallist):
    for model in evallist:
        print(model[0], " : ", model[3])

## training different models
import time
# linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() # definition of the model
linreg_start = time.time() # record starting time
scores_linreg = cross_val_score(lin_reg, X_train_prepared,y_cnt_train, cv = 10, scoring ="neg_mean_squared_error") # computation of the scores from cross-validation
linreg_stop = time.time() # record stopping time
scores_rmse_linreg = np.sqrt(-scores_linreg) # calculation of the root-mean-square-deviation
eval_list.append((lin_reg,scores_rmse_linreg.mean(),scores_rmse_linreg.std(), (linreg_stop - linreg_start)/10)) # include model evaluation in eval_list

# Ridge regression
from sklearn.linear_model import Ridge
lin_ridge = Ridge(alpha=2) # definition of the model
linridge_start = time.time() # record starting time
scores_linridge = cross_val_score(lin_ridge, X_train_prepared,y_cnt_train, cv = 10 , scoring= "neg_mean_squared_error") # computation of the scores from cross-validation
linridge_stop = time.time() # record stopping time
scores_rmse_linridge = np.sqrt(-scores_linridge) # calculation of the root-mean-square-deviation
eval_list.append((lin_ridge, scores_rmse_linridge.mean(),scores_rmse_linridge.std(), (linridge_stop - linridge_start)/10)) # include model evaluation in eval_list

# elastic net
from sklearn.linear_model import ElasticNet
lin_net = ElasticNet(alpha=0.1,l1_ratio=0.5) # definition of the model
linnet_start = time.time() # record starting time
scores_linnet = cross_val_score(lin_net,X_train_prepared,y_cnt_train, cv = 10 , scoring= "neg_mean_squared_error") # computation of the scores from cross-validation
linnet_stop = time.time() # record stopping time
scores_rmse_linnet = np.sqrt(-scores_linnet) # calculation of the root-mean-square-deviation
eval_list.append((lin_net, scores_rmse_linnet.mean(),scores_rmse_linnet.std(), (linnet_stop - linnet_start)/10)) # include model evaluation in eval_list

# support vector machines
from sklearn.svm import SVR
svm_reg = SVR(gamma="auto") # definition of the model
svm_start = time.time() # record starting time
scores_svm = cross_val_score(svm_reg,X_train_prepared,y_cnt_train, cv = 10 , scoring= "neg_mean_squared_error") # computation of the scores from cross-validation
svm_stop = time.time() # record stopping time
scores_rmse_svm = np.sqrt(-scores_svm) # calculation of the root-mean-square-deviation
eval_list.append((svm_reg, scores_rmse_svm.mean(),scores_rmse_svm.std(), (svm_stop - svm_start)/10)) # include model evaluation in eval_list

# decision tree model
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor() # definition of the model
tree_start = time.time() # record starting time
scores_tree = cross_val_score(tree_reg,X_train_prepared,y_cnt_train, cv = 10 , scoring= "neg_mean_squared_error") # computation of the scores from cross-validation
tree_stop = time.time() # record stopping time
scores_rmse_tree = np.sqrt(-scores_tree) # calculation of the root-mean-square-deviation
eval_list.append((tree_reg, scores_rmse_tree.mean(),scores_rmse_tree.std(), (tree_stop - tree_start)/10 )) # include model evaluation in eval_list

# random forest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators = 30) # definition of the model
forest_start = time.time() # record starting time
scores_forest = cross_val_score(forest_reg, X_train_prepared,y_cnt_train, cv = 10 , scoring= "neg_mean_squared_error") # computation of the scores from cross-validation
forest_stop = time.time() # record stopping time
scores_rmse_forest = np.sqrt(-scores_forest) # calculation of the root-mean-square-deviation
eval_list.append((forest_reg, scores_rmse_forest.mean(), scores_rmse_forest.std(), (forest_stop - forest_start)/10)) # include model evaluation in eval_list

# MLP model
from tensorflow import keras # library for modeling multi layer perceptron models
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state=42) # function for k-fold cross validation
scores_mlp = [] # list to store the scores for model evaluation
from sklearn.metrics import mean_squared_error # function for calculating the mean squared error, needed for MLP model evaluation

mlp_start = time.time()
for train, test in kfold.split(X_train_prepared,y_cnt_train.values):
    # create model
    mlp_reg = keras.models.Sequential([
        keras.layers.Dense(30,activation="sigmoid", input_shape=X_train_prepared[train].shape[1:]),
        keras.layers.Dense(30, activation = "sigmoid"),
        keras.layers.Dense(1)
    ])
    mlp_reg.compile(loss="mean_squared_error", optimizer="sgd") # compilation of the model
    mlp_reg.fit(X_train_prepared[train], y_cnt_train.values[train], epochs=20, validation_data = (X_train_prepared[test],y_cnt_train.values[test])) # train model
    y_cnt_pred = mlp_reg.predict(X_train_prepared[test]) # make predictions on the test set
    # calculation of the root-mean-square-deviation
    mlp_reg_mse = mean_squared_error(y_cnt_train.values[test], y_cnt_pred)
    mlp_reg_rmse = np.sqrt(mlp_reg_mse)
    scores_mlp.append(mlp_reg_rmse) # store evaluation score
mlp_stop = time.time()
scores_mlp = np.asarray(scores_mlp) # convert list to numpy array for calculation of mean and standard deviation
eval_list.append((mlp_reg, scores_mlp.mean(),scores_mlp.std(), (mlp_stop - mlp_start)/10))

# print results
print_eval(eval_list) # performance measures
print_time(eval_list) # execution time


## fine-tuning the random forest regressor
from sklearn.model_selection import GridSearchCV

# defining the grid of parameters to be tested
param_grid = [
    {"n_estimators": [30,50,100], "max_features": [6,8,10,12]},
    {"bootstrap": [False], "n_estimators": [30,50,100], "max_features": [6,8,10,12]}
]

# performing grid-search
forestgrid_start = time.time() # record starting time
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_train_prepared, y_cnt_train)
forestgrid_stop = time.time() # record stoping time
# best parameters after grid-search
grid_search.best_params_

# final model
model_bikesharing = grid_search.best_estimator_

# evaluation of the final model on the test set
X_test_prepared = dataprep_pipeline.fit_transform(X_test) # preprocessing of the test set
y_cnt_pred_final = model_bikesharing.predict(X_test_prepared) # predictions of the final model on the test set

final_mse = mean_squared_error(y_cnt_test, y_cnt_pred_final)
final_rmse = np.sqrt(final_mse)

# calculation of the mean absolute error
from sklearn.metrics import mean_absolute_error
mean_absolute_deviation = mean_absolute_error(y_cnt_test,y_cnt_pred_final)
forestgrid_time = forestgrid_stop - forestgrid_start
print("The best model: ", model_bikesharing, ".")
print("The mean absolute_deviation is ", mean_absolute_deviation, ".")
print("Thank you for going through this code!")