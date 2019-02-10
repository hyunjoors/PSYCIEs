from EstimatorSelectionHelper import EstimatorSelectionHelper

import numpy as np
import numpy.random as rand
import pandas as pd
import sklearn.metrics
import csv

from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, Lasso, SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

random_seed = 8424
# train_test_split - test_size
# random_seed


def prepareData(filePath):
    # Training Data Setup
    data = pd.read_csv(filePath)

    X = data.iloc[:, 1:6]
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

    # put all five responses into one "paragraph"
    X = X.stack().groupby(level=0).apply(' '.join)

    return (X, y)


def predict_OCEAN(X_test, y_text, X_dev):

    # list of dictionaries
    parameter_dict = {
        'vect__ngram_range': [(1, 2)],  # (1, 1), (1, 3), (2, 2), ...
        'vect__stop_words': [None],
        'tfidf__use_idf': [True],
        #'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),
        'svd__random_state': [random_seed],
        'svd__n_components': [1],  # np.arange(1,51,2)),
        'clf__kernel': ['rbf'],
        'clf__C': [1.0],
        'clf__gamma': [0.001],
    }

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('svd', TruncatedSVD()),
        ('clf', SVR()),
    ])

    file = "siop_ml_dev_submission_format.csv"
    result = pd.read_csv(file, header=0)
    #print(result.columns, resu)

    gridSearch = GridSearchCV(pipeline, parameter_dict, n_jobs=None, cv=3, verbose=1, scoring='r2')

    for trait in ['O', 'C', 'E', 'A', 'N']:
        print("Predicting Score for %s" % trait)
        gridSearch.fit(X_train, y_train[trait])

        y_pred = pd.DataFrame(gridSearch.predict(X_dev))
        header = trait + '_Pred'
        y_pred.columns = [header]

        result.update(y_pred)
    
    result.to_csv(file, index=False)

if __name__ == "__main__":
    #Enter csv filePath/fileName: asdf
    #Enter random_seed: 0
    #Enter test_size(0-1): 0

    X_train, y_train = prepareData(
        'training_data_participant/siop_ml_train_participant.csv')
    X_dev, y_dev = prepareData(
        'dev_data_participant/siop_ml_dev_participant.csv')


    predict_OCEAN(X_train, y_train, X_dev)
