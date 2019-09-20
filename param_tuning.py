from EstimatorSelectionHelper import EstimatorSelectionHelper
# from processData import processData

import numpy as np
import numpy.random as rand
import pandas as pd
import sklearn.metrics
import csv
import xgboost as xgb
import json

from pandas import Series
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from xgboost import XGBRegressor


random_seed = 42 # Guide to Galaxy


def test_split(filePath, seed, test_size, group, question):
  # Training Data Setup
  data_train = pd.read_csv(filePath)
  questions = pd.read_csv("Questions.csv")

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
    if (question == True):
      # add questions to all entry
      X = X.apply(lambda x: "{} {}".format(question, x))
  else:
    if (question == True):
      # add questions to all entry
      for trait in ['O', 'C', 'E', 'A', 'N']:
        X[trait] = X[trait].apply(
            lambda x: "{} {}".format(questions[trait], x))

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size, shuffle=True)

  return (X_train, X_test, y_train, y_test)



def param_tuning(X_train, X_test, y_train, y_test, group, test_size, question):
  sub_parameter_dict = {  # parameters for vect, tfidf, svd
      # 'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
      # 'tfidf__stop_words': [None, 'english'],
      # 'tfidf__use_idf': [True, False],
      # 'tfidf__max_df': (0, 0.25, 0.5, 1.0),
      # 'tfidf__min_df': (0, 0.25, 0.5, 1.0),
      # 'svd__random_state': [random_seed],
      # 'svd__n_components': [5, 40, 50, 100], # For LSA, a value of 100 is recommended.
    #5, 20, 40, 60, 80,
      # excluded 1 because decomposing down to 1 is non-sense in analyzing the words
  }
  parameter_dict = {
      'LinearRegression': {
          # 'clf__fit_intercept': [True, False],
          # 'clf__normalize': [True, False],
      },
      'SVR': {
          # 'clf__kernel': ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'],
          # 'clf__degree': [2, 3],
          # 'clf__C': np.logspace(-2, 5, 8),
          # 'clf__gamma': list(np.logspace(-3, 2, 6)),
          # 'clf__coef0': [0, 5],
          # #'clf__tol': [],
          # 'clf__epsilon': list(np.logspace(-5, -1, 6)),
          # 'clf__shrinking': [True, False],
      },
      'XGB': {
          # 'clf__max_depth': [3, 6, 10],
          # 'clf__learning_rate': [0.01, 0.1, 0.3],
          # # 'clf__n_estimators': [],
          # # 'clf__silent': [],
          # # ‘reg:logistic’ label must be in [0,1] for logistic regression
          # # count:poisson needs max_delta_step
          # # rank:ndcg sometime gives no result
          # # rank:map --> ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
          # #'reg:linear', 'count:poisson', 'survival:cox', 'rank:pairwise', 'reg:gamma',  'reg:tweedie'
          # 'clf__objective': ['count:poisson', 'survival:cox', 'rank:pairwise', 'reg:gamma',  'reg:tweedie'],
          # 'clf__booster': ['gbtree', 'gblinear', 'dart'],
          # # 'clf__gamma': [0],  # Needs to be tuned
          # # 'clf__min_child_weight': [1],
          # 'clf__max_delta_step': [0, 0.7, 4, 10],
          # 'clf__subsample': [0.5, 1],
          # 'clf__colsample_bytree': [0.5, 1],
          # # #'clf__colsample_bylevel': [], # subsample & bytree will do the job
          # # # can be used in case of very high dimensionality
          # # 'clf__reg_alpha': [0],
          # # 'clf__reg_lambda': [1],  # can be used to reduce overfitting
          # # # can be used in case of high imbalance as it helps in faster convergence
          # # 'clf__sacle_pos_weight': [1],
          # # 'clf__base_score': [],
          # # 'clf__seed': [],
          # # 'clf__random_state': [],
          # # 'clf__missing': [],
          # # 'clf__importance_type': [],
      },
  }

  for key in parameter_dict.keys():
    parameter_dict[key].update(list(sub_parameter_dict.items()))

  clf_dict = {
      'LinearRegression': LinearRegression(),
      'SVR': SVR(),
      'XGB': XGBRegressor(),
  }

  for trait in ['O', 'C', 'E', 'A', 'N']:
    print("Hyper-Parameter Tuning for %s" % trait)
    gridSearch = EstimatorSelectionHelper(clf_dict, parameter_dict)
    if group == 'group':
      gridSearch.tune(X_train, y_train[trait], X_test, y_test[trait],
                      cv = 5, n_jobs=1, verbose=1, scoring='r2', return_train_score=False, error_score='raise', iid=True)
    else:
      gridSearch.tune(X_train[trait], y_train[trait], X_test[trait], y_test[trait],
                      cv=5, n_jobs=1, verbose=1, scoring='r2', return_train_score=False, error_score='raise', iid=True)

    result = []
    result.append({'estimator': gridSearch.best_['estimator']})
    result.append({'trait': trait})
    result.append({'grouped': group})
    result.append({'test_size': test_size})
    result.append({'question': question})
    for key, value in gridSearch.best_['params'].items():
      result.append({key: value})
    result.append({'r': gridSearch.best_['r']})
    result = {k: v for d in result for k, v in d.items()}

    file_name = "tuningResult_" + trait + ".json"
    with open(file_name, 'a') as fp:
      json.dump(result, fp, sort_keys=True, indent=4)

if __name__ == "__main__":

  # for y in ['group', 'ind']:
  #     for question in [True, False]:
  #       print("Tuning with test_size={} & grouping={} & question={}".format(0.05, y, question))
  #       X_train, X_test, y_train, y_test = test_split(
  #           'rand_train_data.csv', random_seed, 0.05, y, question)
  #       param_tuning(X_train, X_test, y_train, y_test, y, 0.05, question)

  data_train = pd.read_csv(
      'training_data_participant/siop_ml_train_participant.csv')
  data_test = pd.read_csv('dev_data_participant/siop_ml_dev_participant.csv')

  X_train = data_train.iloc[:, 1:6]
  y_train = data_train.iloc[:, 6:11]
  X_test = data_test.iloc[:, 1:6]
  y_text = data_test.iloc[:, 6:11]

  # rename columns for easier access
  X_train.rename(columns={'open_ended_1': 'A',
                          'open_ended_2': 'C',
                          'open_ended_3': 'E',
                          'open_ended_4': 'N',
                          'open_ended_5': 'O'}, inplace=True)

  y_train.rename(columns={'E_Scale_score': 'E',
                          'A_Scale_score': 'A',
                          'O_Scale_score': 'O',
                          'C_Scale_score': 'C',
                          'N_Scale_score': 'N'}, inplace=True)
  X_test.rename(columns={'open_ended_1': 'A',
                         'open_ended_2': 'C',
                         'open_ended_3': 'E',
                         'open_ended_4': 'N',
                         'open_ended_5': 'O'}, inplace=True)

  y_text.rename(columns={'E_Scale_score': 'E',
                         'A_Scale_score': 'A',
                         'O_Scale_score': 'O',
                         'C_Scale_score': 'C',
                         'N_Scale_score': 'N'}, inplace=True)

  for y in ['ind']:
    for question in [True, False]:
      print("Tuning with test_size={} & grouping={} & question={}".format(0.05, y, question))
      X_train, X_test, y_train, y_test = test_split('training_data_participant/siop_ml_train_participant.csv', random_seed, 0.05, y, question)
      param_tuning(X_train, X_test, y_train, y_test, y, 0.05, question)
