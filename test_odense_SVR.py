# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:09:01 2017

@author: azfv1n8
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pickle

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

import pipeline as pl
from build_data_set import categorical_columns
from feature_extraction import yX_columns
from test_perfect_wfc import prepare_cv_test_data
from mytimer import Timer

# focus on data relevant for day-ahead electricity trading

train_df = pd.read_pickle('data/cleaned/assembled_data/Odense/cv_data.pkl')
test_df = pd.read_pickle('data/cleaned/assembled_data/Odense/test_data.pkl')

#%%
cat_cols = ['weekend', 'hour', 'weekday', 'month']

def test():
    train_yX_df = train_df[yX_columns['Sc5']]
    test_yX_df = test_df[yX_columns['Sc5']]
    
    
    print train_yX_df.columns, test_yX_df.columns
    train_yXdf_w_dummies = pl.categoricals_to_dummies(train_yX_df, categoricals_in_Xdf=cat_cols)
    test_yXdf_w_dummies = pl.categoricals_to_dummies(test_yX_df, categoricals_in_Xdf=cat_cols)
    for col in train_yXdf_w_dummies.columns:
        if col not in test_yXdf_w_dummies.columns:
            test_yXdf_w_dummies[col] = np.zeros(len(test_yXdf_w_dummies))
    
    train_arr_dict = pl.df_to_np_arr_dict(train_yXdf_w_dummies)
    test_arr_dict = pl.df_to_np_arr_dict(test_yXdf_w_dummies)
    
    dummy_columns = pl.get_dummy_columns(train_yXdf_w_dummies, cat_cols)
    dummy_column_ix = pl.get_dummy_col_ix(train_yXdf_w_dummies, dummy_columns)
    
    scaler = pl.StandardScalerIgnoreDummies(dummy_column_ix, StandardScaler())
    
    train_arr_scaled = scaler.fit_transform(train_arr_dict['array'])
    test_arr_scaled = scaler.transform(test_arr_dict['array'])
    
    train_y = train_arr_scaled[:,0]
    train_X = train_arr_scaled[:,1:]
    
    test_y = test_arr_scaled[:,0]
    test_X = test_arr_scaled[:,1:]
    
    model = SVR(C=4.3, gamma=.02)
    
    model.fit(train_X, train_y)
    ypred = model.predict(test_X)

    return ypred, test_y

def save_model_and_scaler_for_online():
    full_df = pd.concat([train_df, test_df])
    yX_df = full_df[yX_columns['Sc5']]
    yXdf_w_dummies = pl.categoricals_to_dummies(yX_df, categoricals_in_Xdf=cat_cols)
    arr_dict = pl.df_to_np_arr_dict(yXdf_w_dummies)
    dummy_columns = pl.get_dummy_columns(yXdf_w_dummies, cat_cols)
    dummy_column_ix = pl.get_dummy_col_ix(yXdf_w_dummies, dummy_columns)
    
    scaler = pl.StandardScalerIgnoreDummies(dummy_column_ix, StandardScaler())
    arr_scaled = scaler.fit_transform(arr_dict['array'])
    
    train_y = arr_scaled[:,0]
    train_X = arr_scaled[:,1:]
    
    model = SVR(C=4.3, gamma=.02)   
    model.fit(train_X, train_y)

    ypred = model.predict(train_X)
    savepath = 'data/results/fitted_models/fjv/'
    with open (savepath + 'fjv_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(savepath + 'fjv_svr_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return ypred, scaler, train_y
    