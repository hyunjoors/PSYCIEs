# http://www.davidsbatista.net/blog/2018/02/23/model_optimization/

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA


class EstimatorSelectionHelper:

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
            print(params)

            #Pipeline the estimators
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('svd', TruncatedSVD()),
                ('clf', model),
            ])
            
            gs = GridSearchCV(pipeline, params, **grid_kwargs)
            gs.fit(X_train, y_train)

            print("\tPredicting for %s." % key)
            y_pred = gs.predict(X_test)
            r = np.corrcoef(y_pred, y_test)[0, 1]
            
            
            if (abs(r) > abs(max_r)):
                self.best_['estimator'] = model
                self.best_['params'] = gs.best_params_
                self.best_['r'] = r
                self.best_['y_pred'] = y_pred
            
            print("Current Best")
            print(self.best_['estimator'])
            print(self.best_['params'])
            print(self.best_['r'])
            
            print('\tTuning for %s Done.' % key)
