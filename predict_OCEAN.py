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


def predict_OCEAN(X_test, y_text, X_dev, OCEAN_model_dict, OCEAN_params_dict):

    file = "siop_ml_dev_submission_format.csv"
    result = pd.read_csv(file, header=0)

    for trait in ['O', 'C', 'E', 'A', 'N']:
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('svd', TruncatedSVD()),
            ('clf', OCEAN_model_dict[trait]),
        ])
        
        gridSearch = GridSearchCV(pipeline, OCEAN_params_dict[trait],
                          n_jobs=None, cv=3, verbose=5, scoring='r2')

        print("Predicting Score for %s" % trait)
        gridSearch.fit(X_train, y_train[trait])

        y_pred = pd.DataFrame(gridSearch.predict(X_dev))
        header = trait + '_Pred'
        y_pred.columns = [header]

        result.update(y_pred)
    
    result.to_csv(file, index=False)

if __name__ == "__main__":

    X_train, y_train = prepareData(
        'training_data_participant/siop_ml_train_participant.csv')
    X_dev, y_dev = prepareData(
        'dev_data_participant/siop_ml_dev_participant.csv')

    OCEAN_model_dict = {
        'O': SVR(),
        'C': SVR(),
        'E': SVR(),
        'A': SVR(),
        'N': SVR(),
    }

    OCEAN_params_dict= {
       'O': {
           'vect__ngram_range': [(2, 3)],  # (1, 1), (1, 3), (2, 2), ...
           'vect__stop_words': ['english'],

           'tfidf__use_idf': [True],
           #'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),

           'svd__random_state': [random_seed],
           'svd__n_components': [5],  # np.arange(1,51,2)),

           'clf__kernel': ['rbf'],
           'clf__C': [0.1],
           'clf__gamma': [100],
       },
       'C': {
           'vect__ngram_range': [(2, 3)],  # (1, 1), (1, 3), (2, 2), ...
           'vect__stop_words': [None],

           'tfidf__use_idf': [True],
           #'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),

           'svd__random_state': [random_seed],
           'svd__n_components': [5],  # np.arange(1,51,2)),

           'clf__kernel': ['rbf'],
           'clf__C': [0.01],
           'clf__gamma': [100],
       },
       'E': {
           'vect__ngram_range': [(1, 2)],  # (1, 1), (1, 3), (2, 2), ...
           'vect__stop_words': ['english'],

           'tfidf__use_idf': [True],
           #'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),

           'svd__random_state': [random_seed],
           'svd__n_components': [5],  # np.arange(1,51,2)),

           'clf__kernel': ['linear'],
           'clf__C': [1],
           'clf__gamma': [0.001],
       },
       'A': {
           'vect__ngram_range': [(2, 3)],  # (1, 1), (1, 3), (2, 2), ...
           'vect__stop_words': ['english'],

           'tfidf__use_idf': [True],
           #'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),

           'svd__random_state': [random_seed],
           'svd__n_components': [5],  # np.arange(1,51,2)),

           'clf__kernel': ['rbf'],
           'clf__C': [0.01],
           'clf__gamma': [100],
       },
       'N': {
           'vect__ngram_range': [(1, 2)],  # (1, 1), (1, 3), (2, 2), ...
           'vect__stop_words': ['english'],

           'tfidf__use_idf': [True],
           #'tfidf__max_df': (0.25, 0.5, 0.75, 1.0),

           'svd__random_state': [random_seed],
           'svd__n_components': [5],  # np.arange(1,51,2)),

           'clf__kernel': ['linear'],
           'clf__C': [1],
           'clf__gamma': [0.001],
       },
    }


    predict_OCEAN(X_train, y_train, X_dev, OCEAN_model_dict, OCEAN_params_dict)
