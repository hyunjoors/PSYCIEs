from EstimatorSelectionHelper import EstimatorSelectionHelper
from processData import processData

import numpy as np
import numpy.random as rand
import pandas as pd
import sklearn.metrics
import csv
import xgboost as xgb

from pandas import Series
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from xgboost import XGBRegressor

random_seed = 8424


def test_split(filePath, seed, test_size, group):
  # Training Data Setup
  data_train = pd.read_csv(filePath)

  X = data_train.iloc[:, 1:6]
  y = data_train.iloc[:, 6:11]

  # rename columns for easier access
  X.rename(columns={'open_ended_1': 'A',
                    'open_ended_2': 'C',
                    'open_ended_3': 'E',
                    'open_ended_4': 'N',
                    'open_ended_5': 'O'}, inplace=True)

  y.rename(columns={'E_Scale_score': 'E',
                    'A_Scale_score': 'A',
                    'O_Scale_score': 'O',
                    'C_Scale_score': 'C',
                    'N_Scale_score': 'N'}, inplace=True)

  if (group == 'group'):
    # put all five responses into one "paragraph"
    X = X.stack().groupby(level=0).apply(' '.join)

#   X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                       random_state=seed,
#                                                       test_size=test_size,
#                                                       shuffle=True)

  #return (X_train, X_test, y_train, y_test)
  return (X, y)


def processData_dev(filePath, group):
    # Training Data Setup
    data = pd.read_csv(filePath)

    X = data.iloc[:, 1:6]
    y = data.iloc[:, 6:11]

    # rename columns for easier access
    X.rename(columns={'open_ended_1': 'A',
                    'open_ended_2': 'C',
                    'open_ended_3': 'E',
                    'open_ended_4': 'N',
                    'open_ended_5': 'O'}, inplace=True)

    if (data.shape[1] > 6):
        y = data.iloc[:, 6:11]
        y.rename(columns={'E_Scale_score': 'E',
                            'A_Scale_score': 'A',
                            'O_Scale_score': 'O',
                            'C_Scale_score': 'C',
                            'N_Scale_score': 'N'}, inplace=True)
    else:
        y = None

    if (group == 'group'):
        # put all five responses into one "paragraph"
        X = X.stack().groupby(level=0).apply(' '.join)

    return (X, y)


def predict_OCEAN(X_train, y_train, X_dev, OCEAN_model_dict, OCEAN_params_dict, group):

    file = "siop_ml_dev_submission_format.csv"
    result = pd.read_csv(file, header=0)

    for trait in ['O', 'C', 'E', 'A', 'N']:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svd', TruncatedSVD()),
            ('clf', OCEAN_model_dict[trait]),
        ])

        print("Predicting Score for %s" % trait)
        gridSearch = GridSearchCV(pipeline, OCEAN_params_dict[trait])
        
        if group == 'group':
            gridSearch.fit(X_train, y_train[trait])
            y_pred = pd.DataFrame(gridSearch.predict(X_dev))
        else:
            pipeline.fit(X_train[trait], y_train[trait])
            y_pred = pd.DataFrame(pipeline.predict(X_dev[trait]))

        
        header = trait + '_Pred'
        y_pred.columns = [header]

        result.update(y_pred)
    
    result.to_csv(file, index=False)

if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = test_split(
    #     'training_data_participant/siop_ml_train_participant.csv', seed=random_seed, test_size=0.05, group='ind')
    X_train, y_train = test_split(
        'training_data_participant/siop_ml_train_participant.csv', seed=random_seed, test_size=0.05, group='group')
    X_dev, y_dev = processData_dev(
        'dev_data_participant/siop_ml_dev_participant.csv', group='group')

    OCEAN_model_dict = {
        'O': XGBRegressor(),
        'C': XGBRegressor(),
        'E': XGBRegressor(),
        'A': XGBRegressor(),
        'N': XGBRegressor(),
    }

    OCEAN_params_dict= {
       'O': {
           'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
           'tfidf__stop_words': [None, 'english'],
           'tfidf__use_idf': [True, False],
           'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
           'svd__random_state': [random_seed],
           # np.arange(1,51,2)),
           'svd__n_components': [1, 5, 10, 40, 50, 60, 70, 80, 90, 100],

           'clf__booster': ['gbtree', 'dart'],
       },
       'C': {
           'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
           'tfidf__stop_words': [None, 'english'],
           'tfidf__use_idf': [True, False],
           'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
           'svd__random_state': [random_seed],
           # np.arange(1,51,2)),
           'svd__n_components': [1, 5, 10, 40, 50, 60, 70, 80, 90, 100],

           'clf__booster': ['gbtree', 'dart'],
       },
       'E': {
           'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
           'tfidf__stop_words': [None, 'english'],
           'tfidf__use_idf': [True, False],
           'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
           'svd__random_state': [random_seed],
           # np.arange(1,51,2)),
           'svd__n_components': [1, 5, 10, 40, 50, 60, 70, 80, 90, 100],

           'clf__booster': ['gbtree', 'dart'],
       },
       'A': {
           'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
           'tfidf__stop_words': [None, 'english'],
           'tfidf__use_idf': [True, False],
           'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
           'svd__random_state': [random_seed],
           # np.arange(1,51,2)),
           'svd__n_components': [1, 5, 10, 40, 50, 60, 70, 80, 90, 100],

           'clf__booster': ['gbtree', 'dart'],
       },
       'N': {
           'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
           'tfidf__stop_words': [None, 'english'],
           'tfidf__use_idf': [True, False],
           'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
           'svd__random_state': [random_seed],
           # np.arange(1,51,2)),
           'svd__n_components': [1, 5, 10, 40, 50, 60, 70, 80, 90, 100],

           'clf__booster': ['gbtree', 'dart'],
       },
    }

    predict_OCEAN(X_train, y_train, X_dev, OCEAN_model_dict, OCEAN_params_dict, 'group')
