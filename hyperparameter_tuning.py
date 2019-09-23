# September 23, 2019
# hyperparameter_tuning will collect the feature matrices from feature_extraction and
#######################################################################################################################


from feature_extraction import feature_selection
from pandas import Series
from sklearn import datasets
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor
import csv
import numpy as np
import numpy.random as rand
import os
import pandas as pd
import sklearn.metrics
import sys
import xgboost as xgb


if __name__ == "__main__":
