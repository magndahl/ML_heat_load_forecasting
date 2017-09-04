# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 13:50:16 2017

@author: azfv1n8
"""

""
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

from mytimer import Timer

# focus on data relevant for day-ahead electricity trading
    
df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
yX_columns = OrderedDict(
               [('Sc1', ['prod',
                         'prod_lag24or48',
                         'prod_lag168',
                         'Tout',
                         'Tout_lag4',
                         'vWind',
                         'sunRad',
                         'sunRad_lag4']), 
                ('Sc2', ['prod',
                         'prod_lag24or48',
                         'prod_lag168',
                         'Tout',
                         'Tout_lag4',
                         'vWind',
                         'sunRad',
                         'sunRad_lag4',
                         'weekend',
                         'hour',
                         'weekday',
                         'month']),
                ('Sc3', ['prod',
                         'prod_lag24or48',
                         'prod_lag168',
                         'Tout',
                         'Tout_lag4',
                         'vWind',
                         'sunRad',
                         'sunRad_lag4',
                         'weekend',
                         'observance',
                         'national_holiday',
                         'school_holiday',
                         'hour',
                         'weekday',
                         'month'])])
          
        
        
        
def main():
    scenarios = ['Sc%i'%i for i in (1,2,3)]
    Xs, ys, scalers = build_scale_datasets(scenarios)
    
    regressors = OrderedDict([('OLS', LinearRegression()),
                  ('SVR', SVR(C=4.3, gamma=.02)),
                  ('MLP', MLPRegressor(hidden_layer_sizes=(110,), alpha=0.1, random_state=1, solver='adam', max_iter=400))])
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
    
    with open('data/results/cross_vali_3sc_3models.pkl', 'wb') as f:
        pickle.dump(result_scores, f)
    
            
def build_scale_datasets(scenarios=yX_columns.keys()):
    Xs = {}
    ys = {}
    scalers = {}
    for scenario in scenarios:
        yX_df = df[yX_columns[scenario]]
        categoricals_in_yX_df = [c for c in categorical_columns if c in yX_df.columns]
        yXdf_w_dummies = pl.categoricals_to_dummies(yX_df, categoricals_in_yX_df)
        
        arr_dict = pl.df_to_np_arr_dict(yXdf_w_dummies)
        
        dummy_columns = pl.get_dummy_columns(yXdf_w_dummies, categorical_columns)
        dummy_column_ix = pl.get_dummy_col_ix(yXdf_w_dummies, dummy_columns)
        
        scaler = pl.StandardScalerIgnoreDummies(dummy_column_ix, StandardScaler())
        scalers[scenario] = scaler
        arr_scaled = scaler.fit_transform(arr_dict['array'])
        
        ys[scenario] = arr_scaled[:,0]
        Xs[scenario] = arr_scaled[:,1:]
    
    return Xs, ys, scalers    
        
        
if __name__=='__main__':
    main()
    
