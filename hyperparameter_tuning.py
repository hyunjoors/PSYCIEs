#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Monday, ‎September ‎23, ‎2019, ‏‎12:58:27 PM
Last Modified on Mon Nov  4 13:31:44 2019

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





###############################################################################################################
# Process data with features and return Z_var_list, which will be used as 'X' in hyper parameter tuning
###############################################################################################################
def import_features(features_dict):
    features = pd.DataFrame()
    features['bag_of_word'] = fs.bag_of_word(*features_dict['bag_of_word'])
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
def feature_eval(train_data):
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
# Hyper-parameter tuning with the selected features
###############################################################################################################
def tune(train_data, dev_data, selected_feature):
    best_hyperparameters = []
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
        'readability': {},
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
    selected_feature = feature_eval(train_data)

    best_hyperparameter_list = tune(train_data, dev_data, selected_feature)
    
