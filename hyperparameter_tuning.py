# September 23, 2019
# hyperparameter_tuning will collect the feature matrices from feature_extraction and
#######################################################################################################################

from feature_extraction import feature_selection
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
    bag_of_word = features.bag_of_word(*features_dict['bag_of_word'])
    # doc2vec = features.doc2vec(*features_dict['doc2vec'])
    # dtm = features.dtm(*features_dict['dtm'])
    # sentiment = features.sentiment_analysis
    # (*features_dict['sentiment_analysis'])
    # ELMo = features.ELMo(*features_dict['ELMo'])
    # lexi = features.lexical_diversity(*features_dict['lexical_diversity'])
    # readability = features.readability(*features_dict['readability'])
    # topic = features.topic_modeling(*features_dict['topic_modeling'])
    print(bag_of_word)
    Z_var_list = []

    return Z_var_list

###############################################################################################################
# Hyper-parameter tune
###############################################################################################################
def tune(Z_var_list, X, y):
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
    features = feature_selection(full_data.clean_text, features_dict)
    var_list = import_features(features_dict)
    
    full_data['clean_text'] = full_data.full_text.apply(clean_text)
    train_data = full_data.clean_text[0:1088]
    dev_data = full_data.clean_text[1088:1388]
    test_data = full_data.clean_text[1388:1688]
    best_hyperparameter_list = tune(var_list, X, y)
