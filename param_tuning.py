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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from xgboost import XGBRegressor


random_seed = 8424

def processData(filePath, seed, test_size, group):
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

def param_tuning_group(X_train, X_test, y_train, y_test):

  sub_parameter_dict = {  # parameters for vect, tfidf, svd
      'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],  # (1, 1), (1, 3), (2, 2), ...
      'vect__stop_words': [None, 'english'],
      'tfidf__use_idf': [True],
      #'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
      'svd__random_state': [random_seed],
      'svd__n_components': [1, 5, 10, 40, 50],  # np.arange(1,51,2)),
  }

  # list of dictionaries
  parameter_dict = {
    'LinearRegression': {
      'clf__fit_intercept': [True, False],
      'clf__normalize': [True, False],
      },
    'SVR': {
      'clf__kernel': ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'],
      'clf__degree': [2, 3],
      'clf__C': np.logspace(-2, 5, 8),
      'clf__gamma': list(np.logspace(-3, 2, 6)),
      'clf__coef0': [0, 5],
      #'clf__tol': [],
      'clf__epsilon': list(np.logspace(-5, -1, 6)),
      'clf__shrinking': [True, False],
      },
    'XGB': {
      'clf__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
      'clf__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
      'clf__n_estimators': [],
      'clf__silent': [],
      'clf__objective': ['binary:logistic', 'multi:softmax'],
      'clf__booster': ['gbtree', 'gblinear'],
      'clf__gamma': [0], ## Needs to be tuned
      'clf__min_child_weight': [1], ##
      #'clf__max_delta_step': [],
      'clf__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
      'clf__colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
      #'clf__colsample_bylevel': [], # subsample & bytree will do the job
      'clf__reg_alpha': [0], # can be used in case of very high dimensionality
      'clf__reg_lambda': [1], # can be used to reduce overfitting
      'clf__sacle_pos_weight': [1], # can be used in case of high imbalance as it helps in faster convergence
      'clf__base_score': [],
      'clf__seed': [],
      'clf__random_state': [],
      'clf__missing': [],
      'clf__importance_type': [],
      },
    'ElasticNet': {
      'clf__l1_ratio': [],
      'clf__eps': [],
      'clf__n_alphas': [],
      'clf__alphas': [],
      'clf__fit_intercept': [],
      'clf__normalize': [],
      'clf__precompute': [],
      'clf__max_iter': [],
      'clf__tol': [],
      'clf__cv': [],
      'clf__copy_X': [],
      'clf__positive': [],
      'clf__random_state': [],
      'clf__selection': [],
      },
  }

  for key in parameter_dict.keys():
    parameter_dict[key].update(list(sub_parameter_dict.items()))

  clf_dict = {
      'LinearRegression': LinearRegression(),
      'SVR': SVR(),
<<<<<<< HEAD
      'XGBoost',
      'NB': ,
=======
      'XGB': XGBRegressor,
      #'ElasticNet': ElasticNetCV(),
>>>>>>> 9fdb1c408592ec19a212c4cbb0d737271566ff9f
  }

  # relationshp between question and answers
  # most common words and least common word for each answers
  # Doc to Vect / word to vect

  # XGBoosts
  # Naive Bayes
  # based on the OCEAN score
  # calculate idf 
  # # pipeline = Pipeline([
  #               ('vect', CountVectorizer()),
  #               ('tfidf', TfidfTransformer()),
  #               ('svd', TruncatedSVD()),
  #               ('clf', model),
  #           ])

  
  for trait in ['O', 'C', 'E', 'A', 'N']:
    print("Hyper-Parameter Tuning for %s" % trait)
    gridSearch = EstimatorSelectionHelper(clf_dict, parameter_dict)
    gridSearch.tune(X_train, y_train[trait], X_test, y_test[trait],
                    n_jobs=-5, cv=3, verbose=1, return_train_score=False)
    
    result = []
    result.append({'trait': trait})
    result.append({'estimator': gridSearch.best_['estimator']})
    for key, value in gridSearch.best_['params'].items():
      result.append({key: value})
    result.append({'r': gridSearch.best_['r']})
    #result = {k: v for d in result for k, v in d.items()}
    result = dict([(k, Series(v)) for d in result for k, v in d.items()])
    print(result)
    result = pd.DataFrame(result)
    
    print('Best Hyper-Parameter Result for trait {}\n{}\n'.format(trait, result))

    result.to_csv("tuningResult_group.csv", mode='a', index=False)
    
def param_tuning_individual(X_train, X_test, y_train, y_test):

  sub_parameter_dict = {  # parameters for vect, tfidf, svd
      'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
      'vect__stop_words': [None, 'english'],
      'tfidf__use_idf': [True],
      'svd__random_state': [random_seed],
      'svd__n_components': [1, 5],
  }

  # list of dictionaries
  parameter_dict = {
      'SVR': {
          'clf__kernel': ['rbf', 'linear'],
          'clf__C': np.logspace(-2, 5, 8),
          'clf__gamma': list(np.logspace(-3, 2, 6)),
      },
      'LinearRegression': {

      },
  }

  for key in parameter_dict.keys():
    parameter_dict[key].update(list(sub_parameter_dict.items()))

  clf_dict = {
      'LinearRegression': LinearRegression(),
      'SVR': SVR(),
  }

  for trait in ['O', 'C']:#, 'E', 'A', 'N']:
    print("Hyper-Parameter Tuning for %s" % trait)
    gridSearch = EstimatorSelectionHelper(clf_dict, parameter_dict)
    gridSearch.tune(X_train[trait], y_train[trait], X_test[trait], y_test[trait],
                    n_jobs=None, cv=3, verbose=1, return_train_score=False)

    result = []
    result.append({'trait': trait})
    result.append({'estimator': gridSearch.best_['estimator']})
    for key, value in gridSearch.best_['params'].items():
      result.append({key: value})
    result.append({'r': gridSearch.best_['r']})
    result = {k: v for d in result for k, v in d.items()}
    result = pd.DataFrame(result)

    print('Best Hyper-Parameter Result for trait {}\n{}\n'.format(trait, result))

    result.to_csv("tuningResult_individual.csv", mode='a', index=False)



if __name__ == "__main__":
  #Enter csv filePath/fileName: asdf
  #Enter random_seed: 0
  #Enter test_size(0-1): 0

  # X_train, X_test, y_train, y_test = processData(
  #     'training_data_participant/siop_ml_train_participant.csv', random_seed, 0.25, 'group')
  # param_tuning_group(X_train, X_test, y_train, y_test)
  
  # 'individual' has an issue with ValueError: empty vocabulary; perhaps the documents only contain stop words
  # because the individual's document has only one string.

  print("Tuning with test_size=0.05")
  X_train, X_test, y_train, y_test = processData(
      'training_data_participant/siop_ml_train_participant.csv', random_seed, 0.05, 'group')
  #param_tuning_individual(X_train, X_test, y_train, y_test)
  param_tuning_group(X_train, X_test, y_train, y_test)

  print("Tuning with test_size=0.1")
  X_train, X_test, y_train, y_test = processData(
      'training_data_participant/siop_ml_train_participant.csv', random_seed, 0.1, 'group')
  #param_tuning_individual(X_train, X_test, y_train, y_test)
  param_tuning_group(X_train, X_test, y_train, y_test)

  print("Tuning with test_size=0.25")
  X_train, X_test, y_train, y_test = processData(
      'training_data_participant/siop_ml_train_participant.csv', random_seed, 0.25, 'group')
  #param_tuning_individual(X_train, X_test, y_train, y_test)
  param_tuning_group(X_train, X_test, y_train, y_test)
