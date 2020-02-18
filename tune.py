#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Monday, ‎September ‎23, ‎2019, ‏‎12:58:27 PM
Last Modified on 2/18/2020

@author: Hyun Joo Shin
"""

from GridSearch import GridSearch
from keras import backend as K 
from keras.engine import Layer
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, MultiTaskElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor
import csv
import keras.layers as layers
import nltk
import numpy as np
import os
import pandas as pd
import re
import sys
import xgboost as xgb




###############################################################################################################
# Import the process data from R Studio
###############################################################################################################
def import_features(features_dict):
    df = pd.read_csv("features.csv")
    return df


###############################################################################################################
# Hyper-parameter tuning (with the selected features)
###############################################################################################################
# def tune(train_data_X, train_data_y, dev_data_X, dev_data_y, selected_feature):
def tune(train_data, dev_data):
    
    best_hyperparameters = []
    clf_dict = {
        'neural': KerasRegressor(),
        'forest': RandomForestRegressor(),
        'ridge': Ridge(),
        'elastic': ElasticNet(),
    }
    param_list = {
        'neural': {
            'clf__build_fn': [lambda: ElmoRegressionModel(**model_params)],
            'clf__epochs': [10],
            'clf__batch_size': [32],
            'clf__verbose': [1]
        },
        'forest': {
            'clf__n_estimators': [100],
            'clf__criterion': ['mse'],
            'clf__max_depth': [None],
            'clf__min_samples_split': [2],
            'clf__min_samples_leaf': [1],
            'clf__min_weight_fraction_leaf': [0.0],
            'clf__max_features': ['auto'],
            'clf__max_leaf_nodes': [None],
            'clf__min_impurity_decrease': [0.0],
            'clf__min_impurity_split': [None],
            'clf__bootstrap': [True],
            'clf__oob_score': [False],
            'clf__n_jobs': [None],
            'clf__random_state': [None],
            'clf__verbose': [0],
            'clf__warm_start': [False],
            'clf__ccp_alpha': [0.0],
            'clf__max_samples': [None]
        },
        'ridge': {
            'clf__alpha': [1.0], 
            'clf__fit_intercept': [True], 
            'clf__normalize': [False], 
            'clf__copy_X': [True],
            'clf__max_iter': [None],
            'clf__tol': [0.001],
            'clf__solver': ['auto'],
            'clf__random_state': [None],
        },
        'elastic': {
            'clf__alpha': [1.0],
            'clf__l1_ratio': [0.5],
            'clf__fit_intercept': [True],
            'clf__normalize': [False],
            'clf__precompute': [False],
            'clf__max_iter': [1000],
            'clf__copy_X': [True],
            'clf__tol': [0.0001],
            'clf__warm_start': [False],
            'clf__positive': [False],
            'clf__random_state': [None],
            'clf__selection': ['cyclic']
        },
    }

    for key in parameter_list.keys():
        print('Running Parameter TUning for %s' % key)
        
        pipe = Pipeline([('clf', clf_dict.get(key))])
        
        gs = GridSearchCV(pipe, param_list.get(key), cv=5, n_jobs=1, verbose=1, scoring='r2', return_train_score=False, error_score='raise', iid=True)
        gs.fit(train_data, dev_data)
    return best_hyperparameters





if __name__ == "__main__":
    
    data = pd.read_csv("mega_dataset.csv")
    
    train_data = data.loc["Dataset" == "Train"]
    dev_data = data.loc["Dataset" == "Dev"]
    test_data = data.loc["Dataset" == "Test"]
    

    # best_hyperparameter_list = tune(train_data_X, train_data_y, dev_data_X, dev_data_y, selected_feature)
    best_hyperparameter_list = tune(train_data, dev_data)
    
