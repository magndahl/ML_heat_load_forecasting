# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 10:44:07 2017

@author: azfv1n8
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pickle


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

import pipeline as pl
from build_data_set import categorical_columns

from mytimer import Timer

#%%

# focus on data relevant for day-ahead electricity trading
def main():
        
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
    
    scaler = pl.StandardScalerIgnoreDummies(dummy_column_ix, StandardScaler(), center_dummies=False)
    
    arr_scaled = scaler.fit_transform(arr_dict['array'])
    
    y = arr_scaled[:,0]
    X = arr_scaled[:,1:]
    
    MLS_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    gbr = GradientBoostingRegressor(n_estimators=1000)
    rfr = RandomForestRegressor(n_estimators=100)
    
    cv_scores = cross_val_score(gbr, X, y, cv=6, scoring=MLS_scorer, verbose=True, n_jobs=-1)
        
    print cv_scores, cv_scores.mean()
    
    
    
    return cv_scores


if __name__=='__main__':
    with Timer('CV'):
        main()