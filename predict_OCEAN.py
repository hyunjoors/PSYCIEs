from EstimatorSelectionHelper import EstimatorSelectionHelper

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

  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      random_state=seed,
                                                      test_size=test_size,
                                                      shuffle=True)

  return (X_train, X_test, y_train, y_test)
  #return (X, y)


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


def predict_OCEAN_dev(X_train, y_train, X_dev, OCEAN_model_dict, OCEAN_params_dict, group):

    file = "siop_ml_dev_submission_format.csv"
    result = pd.read_csv(file, header=0)

    for trait in ['E', 'A']:#, 'C', 'E', 'A', 'N']:
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('svd', TruncatedSVD()),
            ('clf', OCEAN_model_dict[trait]),
        ])

        print("Predicting Score for %s" % trait)
        gridSearch = GridSearchCV(pipeline, OCEAN_params_dict[trait],
                                    n_jobs=1, cv=3, verbose=1, scoring='r2', return_train_score=False, error_score='raise', iid=True)
        
        if group == 'group':
            gridSearch.fit(X_train, y_train[trait])
            y_pred = pd.DataFrame(gridSearch.predict(X_dev))
        else:
            gridSearch.fit(X_train[trait], y_train[trait])
            y_pred = pd.DataFrame(gridSearch.predict(X_dev[trait]))
        
        header = trait + '_Pred'
        y_pred.columns = [header]

        result.update(y_pred)
    
    result.to_csv(file, index=False)


def predict_OCEAN_test(X_train, y_train, X_dev, OCEAN_model_dict, OCEAN_params_dict, group):

    file = "siop_ml_test_submission_format.csv"
    result = pd.read_csv(file, header=0)

    for trait in ['E', 'A']:  # , 'C', 'E', 'A', 'N']:
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('svd', TruncatedSVD()),
            ('clf', OCEAN_model_dict[trait]),
        ])

        print("Predicting Score for %s" % trait)
        gridSearch = GridSearchCV(pipeline, OCEAN_params_dict[trait],
                                  n_jobs=1, cv=3, verbose=1, scoring='r2', return_train_score=False, error_score='raise', iid=True)

        if group == 'group':
            gridSearch.fit(X_train, y_train[trait])
            y_pred = pd.DataFrame(gridSearch.predict(X_dev))
        else:
            gridSearch.fit(X_train[trait], y_train[trait])
            y_pred = pd.DataFrame(gridSearch.predict(X_dev[trait]))

        header = trait + '_Pred'
        y_pred.columns = [header]

        result.update(y_pred)

    result.to_csv(file, index=False)

if __name__ == "__main__":
    OCEAN_model_dict = {
        'O': XGBRegressor(),
        'C': XGBRegressor(),
        'E': SVR(),
        'A': SVR(),
        'N': LinearRegression(),
    }

    OCEAN_params_dict= {
       'O': {
            'tfidf__ngram_range': [(1, 1)],
            'tfidf__stop_words': [None],
            'tfidf__use_idf': [True],
            'tfidf__max_df': (0.1, 1.0),
            'svd__random_state': [random_seed],
            # np.arange(1,51,2)),
            'svd__n_components': [10],

            'clf__booster': ['gbtree'],
            'clf__max_depth': [6],
            #'clf__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'clf__objective': ['rank:pairwise'],#, 'rank:ndcg', 'rank:map', 'reg:gamma', 'reg:tweedie'],
            'clf__booster': ['gbtree', 'gblinear', 'dart'],
            'clf__subsample': [0.5],
            'clf__colsample_bytree': [0.6],
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
           'vect__ngram_range': [(1, 2)],
           'vect__stop_words': ['english'],
            'tfidf__use_idf': [True],
            'svd__random_state': [random_seed],
            # np.arange(1,51,2)),
            'svd__n_components': [50],


            'clf__kernel': ['rbf'],
            'clf__C': [0.1],
            'clf__gamma': [10],
       },
       'A': {
           'vect__ngram_range': [(1, 1)],
           'vect__stop_words': ['english'],
           'tfidf__use_idf': [True],
           'svd__random_state': [random_seed],
           # np.arange(1,51,2)),
           'svd__n_components': [40],

           'clf__kernel': ['rbf'],
           'clf__C': [0.1],
           'clf__gamma': [10],
       },
       'N': {
           'tfidf__ngram_range': [(1, 3)],
           'tfidf__stop_words': ['english'],
           'tfidf__use_idf': [True],
           #'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
           'svd__random_state': [random_seed],
           # np.arange(1,51,2)),
           'svd__n_components': [40],

           # 40	8424	TRUE	1, 3	English
       },
    }

    X_dev, y_dev = processData_dev(
        'dev_data_participant/siop_ml_dev_participant.csv', group='group')
    X_test, y_test = processData_dev(
        'test_data/siop_ml_test_participant.csv', group='group')

    print("with entire training data")
    X_train, X_temp, y_train, y_temp = test_split(
        'training_data_participant/siop_ml_train_participant.csv', seed=random_seed, test_size=0.1, group='group')
    predict_OCEAN_dev(X_train, y_train, X_dev, OCEAN_model_dict, OCEAN_params_dict, 'group')
    predict_OCEAN_test(X_train, y_train, X_test, OCEAN_model_dict, OCEAN_params_dict, 'group')
