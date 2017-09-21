# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:04:22 2017

@author: azfv1n8
"""

import pandas as pd

from collections import OrderedDict
import cPickle as pickle

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

import pipeline as pl
from build_data_set import categorical_columns
from feature_extraction import build_scale_datasets

from mytimer import Timer


df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')

yX_columns = OrderedDict([('Sc4', ['prod',
                         'prod_lag24or48',
                         'prod_lag168',
                         'Tout',
                         'Tout_lag4',
                         'vWind',
                         'weekend',
                         'hour',
                         'weekday',
                         'month'])])

def main():
    scenarios = ['Sc%i'%i for i in (4,)]
    Xs, ys, scalers = build_scale_datasets(scenarios, yX_columns=yX_columns)
    
    regressors = OrderedDict([('OLS', LinearRegression()),
                  ('SVR', SVR(C=4.3, gamma=.02))])
    MLS_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    result_scores = {}
    for scenario in scenarios:
        result_scores[scenario] = {}
        for reg_key in regressors.keys():
            reg = regressors[reg_key]
            X = Xs[scenario]
            y = ys[scenario]
            
            with Timer(scenario + reg_key):
                cv_scores = cross_val_score(reg, X, y, cv=6, verbose=True, n_jobs=4, scoring=MLS_scorer)
            result_scores[scenario][reg_key] = cv_scores
            
            print scenario, reg_key, cv_scores.mean()
    
    with open('data/results/cross_vali_solar_free_models.pkl', 'wb') as f:
        pickle.dump(result_scores, f)
        
        
    
   
     
if __name__=="__main__":
    main()