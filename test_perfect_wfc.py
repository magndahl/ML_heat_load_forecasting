# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:53:32 2017

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

from mytimer import Timer





def main():
    scenarios = ['Sc%i'%i for i in (1,2,3)]
    
    cv_df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')
    test_df = pd.read_pickle('data/cleaned/assembled_data/test_data.pkl')
    cv_Xs, cv_ys, test_Xs, test_ys, scalers = prepare_cv_test_data(scenarios, cv_df, test_df)
    
    regressors = OrderedDict([('OLS', LinearRegression()),
                  ('SVR', SVR(C=4.3, gamma=.02)),
                  ('MLP', MLPRegressor(hidden_layer_sizes=(110,), alpha=0.1, random_state=1, solver='adam', max_iter=400))])
    
    

    test_scenarios(regressors, scenarios, cv_Xs, cv_ys, \
                       test_Xs, test_ys, scalers, save_res=True, save_fit_models=False,\
                       load_regressors=True)
    
    ytrue = test_ys['Sc3']
    
    maes = calc_scenario_rescalced_metrics(ytrue, metric=mean_absolute_error)
    rmses = calc_scenario_rescalced_metrics(ytrue, metric=root_mean_squared_error)
    mapes = calc_scenario_rescalced_metrics(ytrue, metric=mean_absolute_percentage_error)

    return maes, rmses, mapes


def prepare_cv_test_data(scenarios, cv_df, test_df):

    cv_Xs = {}
    cv_ys = {}
    test_ys = {}
    test_Xs = {}
    scalers = {}
    
    for scenario in scenarios:
        yX_cv_df = cv_df[yX_columns[scenario]]
        categoricals_in_yX_df = [c for c in categorical_columns if c in yX_cv_df.columns]
        yX_cv_df_w_dummies = pl.categoricals_to_dummies(yX_cv_df, categoricals_in_yX_df)
        cv_arr_dict = pl.df_to_np_arr_dict(yX_cv_df_w_dummies)
        
        yX_test_df = test_df[yX_columns[scenario]]
        yX_test_df_w_dummies = pl.categoricals_to_dummies(yX_test_df, categoricals_in_yX_df)
        test_arr_dict = pl.df_to_np_arr_dict(yX_test_df_w_dummies)
        
        dummy_columns = pl.get_dummy_columns(yX_cv_df_w_dummies, categorical_columns)
        dummy_column_ix = pl.get_dummy_col_ix(yX_cv_df_w_dummies, dummy_columns)
        
        scaler = pl.StandardScalerIgnoreDummies(dummy_column_ix, StandardScaler())
        scalers[scenario] = scaler
        cv_arr_scaled = scaler.fit_transform(cv_arr_dict['array'])
        test_arr_scaled = scaler.transform(test_arr_dict['array']) # notice that the same scaling is used as before!
        
        cv_ys[scenario] = cv_arr_scaled[:,0]
        cv_Xs[scenario] = cv_arr_scaled[:,1:]
        
        test_ys[scenario] = test_arr_scaled[:,0]
        test_Xs[scenario] = test_arr_scaled[:,1:]
    
    return cv_Xs, cv_ys, test_Xs, test_ys, scalers
    


    
def test_scenarios(regressors, scenarios, cv_Xs, cv_ys, test_Xs, test_ys, scalers, \
                   save_res=True, save_fit_models=True, save_prefix='test_perfect_wfc_',\
                   load_regressors=False):
    result_scores = {}
    ypreds = {}
    for scenario in scenarios:
        print scenario
        result_scores[scenario] = {}
        ypreds[scenario] = {}
        fitX = cv_Xs[scenario]
        fity = cv_ys[scenario]
        testX = test_Xs[scenario]
        testy = test_ys[scenario]
        for reg_key in regressors.keys():
            print reg_key
            if not load_regressors:
                regressor = regressors[reg_key]
                regressor.fit(fitX, fity)
            elif load_regressors:
                try:
                    with open('data/results/fitted_models/%s_%s.pkl'%(scenario, reg_key), 'r') as f:
                        regressor = pickle.load(f)
                except:
                    print 'Loading %s_%s.pkl failed, fitting instead'%(scenario, reg_key)
                    regressor = regressors[reg_key]
                    regressor.fit(fitX, fity)
                
            prediction = regressor.predict(testX)
            ypreds[scenario][reg_key] = prediction
            MSEscore = mean_squared_error(testy, prediction)
            result_scores[scenario][reg_key] = MSEscore
            print MSEscore
            if save_fit_models and not load_regressors:
                with open('data/results/fitted_models/%s_%s.pkl'%(scenario, reg_key), 'wb') as f:
                    pickle.dump(regressor, f)
    
    if save_res:
        with open('data/results/%sMSE.pkl'%save_prefix, 'wb') as f:
            pickle.dump(result_scores, f)
           
        with open('data/results/%spreds.pkl'%save_prefix, 'wb') as f:
            pickle.dump(ypreds, f)
            
        with open('data/results/%sscalers.pkl'%save_prefix, 'wb') as f:
            pickle.dump(scalers, f)
            
    return ypreds, result_scores


def calc_scenario_rescalced_metrics(ytrue, metric=mean_absolute_error, \
                                    preds_dict_path='data/results/test_perfect_wfc_preds.pkl',\
                                    scaler_dict_path='data/results/test_perfect_wfc_scalers.pkl'):
    """ This function rescales the production to the original units (MW) before
        calculating metrics such as MAE, MAPE or RMSE """
        
        
    with open(preds_dict_path, 'r') as f:    
        pred_dict = pickle.load(f)
        
    with open(scaler_dict_path, 'r') as f:
        scaler_dict = pickle.load(f)
    
    
    result_scores = {}
    for scenario in pred_dict.keys():
        scaler = scaler_dict[scenario]
        true_prod = scaler.inverse_transform_y(ytrue)
        result_scores[scenario] = {}
        for reg_key in pred_dict[scenario].keys():
            ypred = pred_dict[scenario][reg_key]
            predicted_prod = scaler.inverse_transform_y(ypred)
            result_scores[scenario][reg_key] = metric(true_prod, predicted_prod)
            
    return result_scores


if __name__=="__main__":
    with Timer('Test perfect forecast'):
        main()
