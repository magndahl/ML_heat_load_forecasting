# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:03:02 2017

@author: azfv1n8
"""
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np



def clean_repack_prod(save=False):
    load_path = 'data/raw_input/Fjvdata/'

    dfs = {}
    years = range(2010, 2018)
    for year in years:
        hl_filename = 'FjvMV-%i.csv' % year
        dfs[year] = pd.read_csv(load_path + hl_filename, delimiter=';', decimal=',', encoding='utf_16_le', index_col='#UTC')    
        
        
    df = pd.DataFrame(index=pd.date_range(dfs[years[0]].index[0], dfs[years[-1]].index[-1], freq='H'), columns=['Value', 'Quality'])
    
    for yr in years:
        for ts in dfs[yr].index:
            try:
                df.at[ts, :] = dfs[yr].loc[ts, :]
            except:
                print "Failed at %s" % ts
                
    
    df.fillna(method='ffill')
    
    if save:
        df.to_pickle('data/cleaned/Odense/prod2010_2017.pkl')

    return df


def clean_repack_weather(save=False):  
    load_path = 'data/raw_input/Fjvdata/Vejr/'
    Toutdfs = {}
    vWinddfs = {}
    years = range(2010, 2018)
    for year in years:
        Tout_filename = 'SV-Udetemperatur-%i.csv' % year
        vWind_filename = 'SV-Vindhastighed-%i.csv'% year
        Toutdfs[year] = pd.read_csv(load_path + Tout_filename, delimiter=';', decimal=',', encoding='utf_16_le', index_col='#UTC')
        vWinddfs[year] = pd.read_csv(load_path + vWind_filename, delimiter=';', decimal=',', encoding='utf_16_le', index_col='#UTC')
                
                
    df = pd.DataFrame(index=pd.date_range(Toutdfs[years[0]].index[0], Toutdfs[years[-1]].index[-1], freq='H'), columns=['Tout', 'ToutQuality', 'vWind', 'VWindQuality'])
        
        
    for yr in years:
        for ts in Toutdfs[yr].index:
            try:
                df.at[ts, ['Tout', 'ToutQuality']] = Toutdfs[yr].loc[ts, ['Value', 'Quality']].as_matrix()
            except:
                print "Failed at Tout %s" % ts
                
        for ts in vWinddfs[yr].index:
            try:
                df.at[ts, ['vWind', 'VWindQuality']] = vWinddfs[yr].loc[ts,['Value', 'Quality']].as_matrix()
            except:
                print "Failed at vWind %s" % ts 
                    
        
        df.fillna(method='ffill')
    
    
    dfclean = df.loc[dt.datetime(2012,4,1,0):, :]
    if save:
        dfclean.to_pickle('data/cleaned/Odense/ToutvWind2012_2017.pkl')
    
    return df


def build_cv_test_dataset(save=False):
    prod_df = pd.read_pickle('data/cleaned/Odense/prod2010_2017.pkl')
    w_df = pd.read_pickle('data/cleaned/Odense/ToutvWind2012_2017.pkl')
    
    ts_start = max(prod_df.index[0], w_df.index[0])
    ts_end = min(prod_df.index[-1], w_df.index[-1])
    
    a_df = pd.DataFrame(index=pd.date_range(ts_start, ts_end, freq='H'), \
                        columns=['prod',
                                 'prod_lag24or48',
                                 'prod_lag168',
                                 'Tout',
                                 'vWind',
                                 'weekend',
                                 'hour',
                                 'weekday',
                                 'month',
                                 'Quality'])
    
    q_df = pd.DataFrame(index=pd.date_range(ts_start, ts_end, freq='H'), columns=['prodQ', 'ToutQ', 'vWindQ'])
    q_df['prodQ'] = prod_df.loc[q_df.index, 'Quality']
    q_df['ToutQ'] = w_df.loc[q_df.index, 'ToutQuality']
    q_df['vWindQ'] = w_df.loc[q_df.index, 'VWindQuality']
       
    a_df['prod'] = prod_df.loc[a_df.index, 'Value']
    a_df['Tout'] = w_df.loc[a_df.index, 'Tout']
    a_df['vWind'] = w_df.loc[a_df.index, 'vWind']
    a_df['Quality'] = q_df.min(axis=1)

    prodlag24 = a_df['prod'].shift(24).fillna(method='bfill')
    prodlag48 = a_df['prod'].shift(48).fillna(method='bfill')
    prod_lag24or48 = [prodlag24[ts] if ts.hour <= 7 else prodlag48[ts] for ts in a_df.index]
    a_df['prod_lag24or48'] = prod_lag24or48
    
    prodlag168 = a_df['prod'].shift(168).fillna(method='bfill')
    a_df['prod_lag168'] = prodlag168
    
    isWeekend = [(ts.weekday_name in ('Saturday', 'Sunday')) for ts in a_df.index]
    a_df['weekend'] = isWeekend
    
    a_df['weekday'] = [ts.dayofweek for ts in a_df.index]
    a_df['hour'] = [ts.hour for ts in a_df.index]
    a_df['month'] = [ts.month for ts in a_df.index]
    
    
    full_df_good_quality = a_df[a_df['Quality']>=192]
    
    split_date = dt.datetime(2017,4,1,0)
    cv_df = full_df_good_quality[full_df_good_quality.index < split_date]
    test_df = full_df_good_quality[full_df_good_quality.index >= split_date]
    
    if save:
        cv_df.to_pickle('data/cleaned/assembled_data/Odense/cv_data.pkl')
        test_df.to_pickle('data/cleaned/assembled_data/Odense/test_data.pkl')
    
    return cv_df, test_df