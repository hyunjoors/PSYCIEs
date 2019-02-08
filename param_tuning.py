# https://stackoverflow.com/questions/50285973/pipeline-multiple-classifiers

from ClfSwitcher import ClfSwitcher
import EstimatorSelectionHelper

import numpy as np
import numpy.random as rand
import pandas as pd
import sklearn.metrics

from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, Lasso, SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

random_seed = 8424
### list of parameters for main
# train_test_split - test_size
# random_seed

def processData(filePath, seed, test_size):
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

  # put all five responses into one "paragraph"
  X = X.stack().groupby(level=0).apply(' '.join)

  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=seed,
                                                    test_size=test_size,
                                                    shuffle=True)

  return (X_train, X_test, y_train, y_test)


def param_tuning(X_train, X_test, y_train, y_test):

  sub_parameter_dict = {  # parameters for vect, tfidf, svd
      'vect__ngram_range': [(1, 2), (2, 3)],  # (1, 1), (1, 3), (2, 2), ...
      'tfidf__use_idf': [True],
      #'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
      'vect__stop_words': ['english', None],
      'svd__random_state': [random_seed],
      'svd__n_components': [1, 5],  # np.arange(1,51,2)),
  }

  # list of dictionaries
  # clf estimator will take a place of ClfSwitcher()
  parameter_dict = [
    # {
    #   # SVM if hinge loss / logreg if log loss
    #   'clf__estimator': [SGDClassifier()],
    #   'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
    #   'clf__estimator__max_iter': [50, 80],
    #   'clf__estimator__tol': [1e-4],
    #   'clf__estimator__loss': ['hinge', 'log', 'modified_huber'],
    # },
    # {
    #   'clf__estimator': [MultinomialNB()],
    #   'clf__estimator__alpha': (1e-2, 1e-3, 1e-1),
    # },
    {
      'clf__estimator': [SVR()],
      'clf__kernel': ['rbf', 'linear'],
      'clf__C': np.logspace(-2, 6, 9),
      'clf__gamma': list(np.logspace(-3, 2, 6)),
    },
    {
      'clf__estimator': [LinearRegression()],
    },
    # {
    #   'clf__estimator': [ExtraTreesClassifier()],
    #   'clf__n_estimators': [16, 32],
    # },
    # {
    #   'clf__estimator': [RandomForestClassifier()],
    #   'clf__n_estimators': [16, 32],
    # },
    # {
    #   'clf__estimator': [AdaBoostClassifier()],
    #   'clf__n_estimators': [16, 32],
    # },
    # {
    #   'clf__estimator': [GradientBoostingClassifier()],
    #   'clf__n_estimators': [16, 32],
    #   'clf__learning_rate': [0.8, 1.0],
    # },
  ]

  for i in range(len(parameter_dict)):
    parameter_dict[i].update(list(sub_parameter_dict.items()))

  # Pipeline the estimators
  pipeline = Pipeline([
      ('vect', CountVectorizer()),
      ('tfidf', TfidfTransformer()),
      ('svd', TruncatedSVD()),
      ('clf', ClfSwitcher()),
  ])

  gridSearch = GridSearchCV(pipeline, parameter_dict, cv=3, verbose=0, return_train_score=False, scoring='r2')
  for trait in ['O', 'C', 'E', 'A', 'N']:
    gridSearch.fit(X_train, y_train[trait])
    y_pred = gridSearch.predict(X_test)
    r = np.corrcoef(y_pred, y_test[trait])[0, 1]
    print('Best Hyper-Parameter Result for trait {}\n Mean r: {}\n{}'.format(trait, r, gridSearch.best_params_.values()))

    result = pd.DataFrame.from_dict(gridSearch.cv_results_)
    result.drop(
        columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params'])
    fileNanme = trait + "_tuning_result.csv"
    result.to_csv(fileNanme, mode='a')


if __name__ == "__main__":
  #Enter csv filePath/fileName: asdf
  #Enter random_seed: 0
  #Enter test_size(0-1): 0

  X_train, X_test, y_train, y_test = processData(
      'training_data_participant/siop_ml_train_participant.csv', random_seed, 0.25)

  param_tuning(X_train, X_test, y_train, y_test)
