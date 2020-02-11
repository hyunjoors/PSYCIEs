#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Monday, ‎September ‎23, ‎2019, ‏‎12:58:27 PM
Last Modified on 2/11/2020

@author: Hyun Joo Shin
"""

from feature_extraction import FeatureExtraction
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
import numpy as np
import os
import pandas as pd
import sys
import xgboost as xgb
import nltk
import re

from keras.layers import Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K 
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer




###############################################################################################################
# Data preparation, pre-processing
# Clean up the texts and prepare&divide the text into [train, dev, text]
###############################################################################################################
full_data = pd.read_csv("./data/2019_siop_ml_comp_data.csv")
# csv data is in order of train(1088) -> dev(300) -> test(300)
full_data['full_text'] = full_data['open_ended_1'] + ' ' + full_data['open_ended_2'] + ' ' + \
                        full_data['open_ended_3'] + ' ' + full_data['open_ended_4'] + ' ' + \
                        full_data['open_ended_5']
full_data['clean_text'] = full_data.full_text.apply(clean_text)
train_data = full_data.clean_text[0:1088]
dev_data = full_data.clean_text[1088:1388]
test_data = full_data.clean_text[1388:1688]




#######################################################################################################################
# Helper functions
# These functions are not the major feature extracting functions. (e.g., clean_text, lemma below)
#######################################################################################################################
# Pre-processing (not removing stopwords, no lemma, etc.)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # only keep english words/letters
    words = text.lower().split()  # lowercase
    
    return ' '.join(words)




###############################################################################################################
# Process data with features and return Z_var_list, which will be used as 'X' in hyper parameter tuning
###############################################################################################################
def import_features(features_dict):
    features = pd.DataFrame()
    features['bag_of_word'] = fs.bag_of_word() # [HYUN] Trying to assign Dataframe to Series
    # features['doc2vec'] = features.doc2vec(*features_dict['doc2vec'])
    # feature['dtm'] = features.dtm(*features_dict['dtm'])
    # features['sentiment'] = features.sentiment_analysis(*features_dict['sentiment_analysis'])
    # features['ELMo'] = features.ELMo(*features_dict['ELMo'])
    # features['lexi_div'] = features.lexical_diversity(*features_dict['lexical_diversity'])
    # features['readability'] = features.readability(*features_dict['readability'])
    # features['topic_model'] = features.topic_modeling(*features_dict['topic_modeling'])
    print(features)
    features.to_csv('./feature_results.csv')
    Z_var_list = []

    return Z_var_list





###############################################################################################################
# Train using single features only, then save the predicted scores & return correlatin for each feature
###############################################################################################################
def feature_tuning(train_data):
    feature_results = {
        'O': {},
        'C': {},
        'E': {},
        'A': {},
        'N': {}
    }
    
    
    print(feature_results)
    return feature_results





###############################################################################################################
# Hyper-parameter tuning (with the selected features)
###############################################################################################################
# def tune(train_data_X, train_data_y, dev_data_X, dev_data_y, selected_feature):
def tune(train_data_X, train_data_y, dev_data_X, dev_data_y):
    
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
            'n_estimators': [100],
            'criterion': ['mse'],
            'max_depth': [None],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'min_weight_fraction_leaf': [0.0],
            'max_features': ['auto'],
            'max_leaf_nodes': [None],
            'min_impurity_decrease': [0.0],
            'min_impurity_split': [None],
            'bootstrap': [True],
            'oob_score': [False],
            'n_jobs': [None],
            'random_state': [None],
            'verbose': [0],
            'warm_start': [False],
            'ccp_alpha': [0.0],
            'max_samples': [None]
        },
        'ridge': {
            'clf__alpha': [1.0], 
            'fit_intercept': [True], 
            'clf__normalize': [False], 
            'clf__copy_X': [True],
            'clf__max_iter': [None],
            'clf__tol': [0.001],
            'clf__solver': ['auto'],
            'clf__random_state': [None],
        },
        'elastic': {
            'alpha': [1.0],
            'l1_ratio': [0.5],
            'fit_intercept': [True],
            'normalize': [False],
            'precompute': [False],
            'max_iter': [1000],
            'copy_X': [True],
            'tol': [0.0001],
            'warm_start': [False],
            'positive': [False],
            'random_state': [None],
            'selection': ['cyclic']
        },
    }

    for key in parameter_list.keys():
        print('Running Parameter TUning for %s' % key)
        
        pipe = Pipeline([('clf', clf_dict.get(key))])
        
        gs = GridSearchCV(pipe, param_list.get(key), cv=5, n_jobs=1, verbose=1, scoring='r2', return_train_score=False, error_score='raise', iid=True)
        gs.fit(train_data_X, train_data_y)
    return best_hyperparameters





if __name__ == "__main__":
    
    # Extract features & combine 
    features_dict = {
        'bag_of_word': {'ngram_range': (1,2),
                        'stop_words': None},
        'doc2vec': {},
        'dtm': {},
        'sentiment_analysis': {},
        'ELMo': {},
        'lexical_diversity': {},
        'readability': {"Flesch.Kincaid",
                        "Dale.Chall.old",
                        "Wheeler.Smith",
                        "meanSentenceLength",
                        "meanWordSyllables",
                        "Strain",
                        "SMOG",
                        "Scrabble",
                        "FOG",
                        "Farr.Jenkins.Paterson",
                        "DRP",
                        "Dale.Chall"
        },
        'topic_modeling': {},
    }

    full_data['clean_text'] = full_data.full_text.apply(clean_text)

    fs = FeatureExtraction(full_data.clean_text, features_dict)
    feature_df = import_features(features_dict)
    pd.concat([full_data, feature_df])
    
    train_data = full_data.clean_text[0:1088]
    dev_data = full_data.clean_text[1088:1388]
    test_data = full_data.clean_text[1388:1688]
    
    # select a number of features or all features
    # selected_feature = feature_tuning(train_data)

    # best_hyperparameter_list = tune(train_data_X, train_data_y, dev_data_X, dev_data_y, selected_feature)
    best_hyperparameter_list = tune(train_data_X, train_data_y, dev_data_X, dev_data_y)
    
