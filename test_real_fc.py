# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:39:17 2017

@author: azfv1n8
"""

from collections import OrderedDict
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error
from custom_metrics import root_mean_squared_error, mean_absolute_percentage_error


from mytimer import Timer
from test_perfect_wfc import prepare_cv_test_data, test_scenarios, calc_scenario_rescalced_metrics


def main():
    scenarios = ['Sc%i'%i for i in (1,2,3,4)]
    cv_df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    test_df = pd.read_pickle('data/cleaned/assembled_data/test_data_real_fc.pkl')
    cv_Xs, cv_ys, test_Xs, test_ys, scalers = prepare_cv_test_data(scenarios, cv_df, test_df)
    
    regressors = OrderedDict([('OLS', LinearRegression()),
                  ('SVR', SVR(C=4.3, gamma=.02)),
                  ('MLP', MLPRegressor(hidden_layer_sizes=(110,), alpha=0.1, random_state=1, solver='adam', max_iter=400))])
    
    test_scenarios(regressors, scenarios, cv_Xs, cv_ys, \
                       test_Xs, test_ys, scalers, save_res=True, save_fit_models=False,\
                       save_prefix='test_real_wfc_', load_regressors=True)
        
        
    
    ytrue = test_ys['Sc3']
    pdp = 'data/results/test_real_wfc_preds.pkl'
    sdp = 'data/results/test_real_wfc_scalers.pkl'
    maes = calc_scenario_rescalced_metrics(ytrue, metric=mean_absolute_error,\
                                           preds_dict_path=pdp, scaler_dict_path=sdp)
    rmses = calc_scenario_rescalced_metrics(ytrue, metric=root_mean_squared_error,\
                                            preds_dict_path=pdp, scaler_dict_path=sdp)
    mapes = calc_scenario_rescalced_metrics(ytrue, metric=mean_absolute_percentage_error,\
                                            preds_dict_path=pdp, scaler_dict_path=sdp)

    return maes, rmses, mapes


def create_forecast_test_dataset():
    forecast_path = 'data/raw_input/DMI ens forecasts/'
    forecast_suffix = '_geo71699_2016010101_to_2017010100.npy'
    
    fcs = {}
    wvars =  ['Tout', 'vWind', 'sunRad']
    for v in wvars:
        fcs[v] = np.load(forecast_path + v + forecast_suffix)
        
    fc_df = pd.DataFrame(fcs, index=pd.date_range(pd.datetime(2016,1,1,1), pd.datetime(2017,1,1,0), freq='H'))
    
    test_df = pd.read_pickle('data/cleaned/assembled_data/test_data.pkl') # load originial test data, with atlas data
    
    ## make substitutions with forecast data!
    for v in wvars:
        test_df[v] = fc_df[v]
        
    fc_lag4 = fc_df.shift(4).fillna(method='backfill')    
    for v in ['Tout', 'sunRad']:
        col = v + '_lag4'
        test_df[col] = fc_lag4[v]
        
    
    test_df.to_pickle('data/cleaned/assembled_data/test_data_real_fc.pkl')
    print "Test data with forecast saved in data/cleaned/assembled_data/test_data_real_fc.pkl"
    
    return



if __name__=="__main__":
    with Timer('Testing with real forecasts'):
        main()