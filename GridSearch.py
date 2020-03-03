import csv
import os
import re
import sys

import keras.layers as layers
import nltk
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow_hub as hub
import xgboost as xgb
from keras import backend as K
from keras.engine import Layer
from keras.layers import (Dense, Dropout, Embedding, Flatten, Input,
                          MaxPooling1D)
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                  MultiTaskElasticNet, Ridge)
from sklearn.model_selection import (GridSearchCV, LeaveOneOut,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor

from GridSearch import GridSearch


#######
#dkjflsdkjf;sd
######
class GridSearch:

    def __init__(self, models_dict, params_dict):
        # if not set(models_dict.keys()).issubset(set(params_dict.keys())):
        #     missing_params = list(set(models.keys()) - set(params_dict.keys()))
        #     raise ValueError(
        #         "Some estimators are missing parameters: %s" % missing_params)
        self.models = models_dict
        self.params = params_dict
        self.keys = models_dict.keys()
        #print(self.keys)
        self.best_ = {
            'estimator': [None],
            'params': {},
            'y_pred': [],
            'r': [],
        }

    def predict(self):
        return self.best_['r']

    def tune(self, X_train, y_train, X_test, y_test, **grid_kwargs):
        max_r = 0
        for key in self.keys:
            print("\tRunning GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]

            #Pipeline the estimators
            pipeline = Pipeline([
                ('clf', model),
            ])

            gs = GridSearchCV(pipeline, params, **grid_kwargs)
            gs.fit(X_train, y_train)

            print("\tPredicting for %s." % key)
            y_pred = gs.predict(X_test)
            r = np.corrcoef(y_pred, y_test)[0, 1]
            print(params)
            print(r)

            if (abs(r) > abs(max_r)):
                self.best_['estimator'] = model
                self.best_['params'] = gs.best_params_
                self.best_['r'] = r
                self.best_['y_pred'] = y_pred

            print("Current Best")
            #print(self.best_['params'])
            print(self.best_['r'])

            print('\tTuning for %s Done.' % key)
