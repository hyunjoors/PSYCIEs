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






#######################################################################################################################
# Helper functions
# These functions are not the major feature extracting functions. (e.g., clean_text, lemma below)
#######################################################################################################################
# Pre-processing (not removing stopwords, no lemma, etc.)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # only keep english words/letters
    words = text.lower().split()  # lowercase
    return ' '.join(words)

# Further clean text using wordnetlemmatizer
lem = WordNetLemmatizer()
def lemma(text):
    words = nltk.word_tokenize(text)
    return ' '.join([lem.lemmatize(w) for w in words])




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




if __name__ == "__main__":

	features = feature_selection(train_data)
	param_dict = {
      'bag_of_word': [],
	  'doc2vec': [],
	  'dtm': [],
	  'sentiment_analysis': [],
	  'ELMo': [],
	  'lexical_diversity': [],
	  'readability': [],
	  'topic_modeling': [],
  }
	bag_of_word = features.bag_of_word(*param_dict['bag_of_word'])
	doc2vec = features.doc2vec(*param_dict['doc2vec'])
	dtm = features.dtm(*param_dict['dtm'])
	sentiment = features.sentiment_analysis(*param_dict['sentiment_analysis'])
	ELMo = features.ELMo(*param_dict['ELMo'])
	lexi = features.lexical_diversity(*param_dict['lexical_diversity'])
	readability = features.readability(*param_dict['readability'])
	topic = features.topic_modeling(*param_dict['topic_modeling'])