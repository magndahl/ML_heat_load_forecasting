# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 13:54:38 2017

@author: azfv1n8
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

import pipeline as pl
from build_data_set import categorical_columns

from mytimer import Timer

#%%
def main():
    
    
    # focus on data relevant for day-ahead electricity trading
    
    df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    yX_columns = ['prod',
                 'prod_lag24or48',
                 'prod_lag168',
                 'Tout',
                 'vWind',
                 'sunRad',
                 'weekend',
                 'observance',
                 'national_holiday',
                 'school_holiday',
                 'hour',
                 'weekday',
                 'month']
    
    yX_df = df[yX_columns]
    yXdf_w_dummies = pl.categoricals_to_dummies(yX_df, categorical_columns)
    
    arr_dict = pl.df_to_np_arr_dict(yXdf_w_dummies)
    
    dummy_columns = pl.get_dummy_columns(yXdf_w_dummies, categorical_columns)
    dummy_column_ix = pl.get_dummy_col_ix(yXdf_w_dummies, dummy_columns)
    
    scaler = pl.StandardScalerIgnoreDummies(dummy_column_ix, StandardScaler())
    
    arr_scaled = scaler.transform(arr_dict['array'])
    
    y = arr_scaled[:,0]
    X = arr_scaled[:,1:]
    
    
    MLS_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    #%% 6 fold cross validation on OLS regression
    OLS_cv_scores = cross_val_score(LinearRegression(), X, y, scoring=MLS_scorer, cv=6)
    OLS_cv_pred = cross_val_predict(LinearRegression(), X, y, cv=6)
    
    
    #%% No calendar info 6 fold cross validation on OLS regression
    X_nocal = arr_scaled[:,1:6]
    
    OLS_cv_scores_nocal = cross_val_score(LinearRegression(), X_nocal, y, scoring=MLS_scorer, cv=6)
    OLS_cv_pred_nocal = cross_val_predict(LinearRegression(), X_nocal, y, cv=6)
    
    #%% SVR experiments
#    
#    SVR_cv_pred = cross_val_predict(SVR(C=1., gamma=.02), X, y, cv=6, verbose=True, n_jobs=4)
#    
#    SVR_cv_err = mean_squared_error(y, SVR_cv_pred)
#    print "SVR_err:", SVR_cv_err
    
    return y, X, scaler #, SVR_cv_pred
    
    
if __name__=="__main__":
    with Timer('new'):
        main()
    