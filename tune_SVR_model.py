# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pickle


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

import pipeline as pl
from mytimer import Timer
from build_data_set import categorical_columns

    #%%

def main():

    # focus on data relevant for day-ahead electricity trading
    df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    yX_columns = ['prod',
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
                 'month']

    yX_df = df[yX_columns]
    yXdf_w_dummies = pl.categoricals_to_dummies(yX_df, categorical_columns)

    arr_dict = pl.df_to_np_arr_dict(yXdf_w_dummies)

    dummy_columns = pl.get_dummy_columns(yXdf_w_dummies, categorical_columns)
    dummy_column_ix = pl.get_dummy_col_ix(yXdf_w_dummies, dummy_columns)

    scaler = pl.StandardScalerIgnoreDummies(dummy_column_ix, StandardScaler())

    
    arr_scaled = scaler.fit_transform(arr_dict['array'])
    

    y = arr_scaled[:,0]
    X = arr_scaled[:,1:]

    MLS_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    param_grid = {'C':[1, 4.3], 'gamma':[.001, 0.01, .02]}
    grid_search_estimator = GridSearchCV(SVR(), param_grid=param_grid, n_jobs=-1, cv=6, scoring=MLS_scorer)
    grid_search_estimator.fit(X,y)
    
    with open('data/svr_gridsearch6pts.pkl', 'wb') as f:
        pickle.dump(grid_search_estimator, f)

    return grid_search_estimator



if __name__=="__main__":
    with Timer('SVR tuning'):
        main()
