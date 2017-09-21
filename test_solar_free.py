# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:09:04 2017

@author: azfv1n8
"""

import pandas as pd
import numpy as np

from collections import OrderedDict
import cPickle as pickle

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from custom_metrics import root_mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

import pipeline as pl
from build_data_set import categorical_columns
from feature_extraction import yX_columns

from test_perfect_wfc import prepare_cv_test_data, test_scenarios, calc_scenario_rescalced_metrics

from mytimer import Timer


def main():
    scenarios = ['Sc%i'%i for i in (4,)]
    
    cv_df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    test_df = pd.read_pickle('data/cleaned/assembled_data/test_data.pkl')
    cv_Xs, cv_ys, test_Xs, test_ys, scalers = prepare_cv_test_data(scenarios, cv_df, test_df)
    
    regressors = OrderedDict([('OLS', LinearRegression()),
                  ('SVR', SVR(C=4.3, gamma=.02)),
                  ('MLP', MLPRegressor(hidden_layer_sizes=(110,), alpha=0.1, random_state=1, solver='adam', max_iter=400))])
    
    

    test_scenarios(regressors, scenarios, cv_Xs, cv_ys, \
                       test_Xs, test_ys, scalers, save_res=True, save_fit_models=False,\
                       load_regressors=True)
    
    ytrue = test_ys['Sc4']
    
    maes = calc_scenario_rescalced_metrics(ytrue, metric=mean_absolute_error)
    rmses = calc_scenario_rescalced_metrics(ytrue, metric=root_mean_squared_error)
    mapes = calc_scenario_rescalced_metrics(ytrue, metric=mean_absolute_percentage_error)

    return maes, rmses, mapes
