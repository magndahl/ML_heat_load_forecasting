# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 15:07:15 2017

@author: azfv1n8
"""

import sys
import sql_tools as sq
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def main(save_cleaned=False):
    plt.close('all')
    print "sc:", save_cleaned
    
    prod2013_2016 = get_2013_2016_prod()
    peak_hour_hist(prod2013_2016)
    plt.title('2013-2016')
        
    pre_fl_prods = {yr:excel_to_series(yr) for yr in (2009, 2010, 2012)}
    
    for yr in (2009, 2010, 2012):
        peak_hour_hist(pre_fl_prods[yr])
        plt.title(str(yr))

    all_prod = assemble_7_year_prod()
    plot_grad_prod_vs_prod(all_prod)
    
    outliers = detect_outliers(all_prod, 125)
    
    cln_prod = clean_outliers_mean(all_prod, outliers)
        
    plot_grad_prod_vs_prod(cln_prod)

    plt.figure()
    plt.plot_date(cln_prod.index, cln_prod, color='b', label='Cleaned prod')
    plt.plot_date(outliers.index, outliers, color='r', label='Worst outliers')
    plt.legend()
    
    if save_cleaned:
        cln_prod.to_pickle('data/cleaned/production2009to2016_not2011.pkl')
    
    return cln_prod


def plot_grad_prod_vs_prod(prod):
    
    plt.figure()
    plt.scatter(prod, prod.diff())
    plt.xlabel('prod')
    plt.ylabel('diff prod')



def assemble_7_year_prod():
    pre_fl_prods = {yr:excel_to_series(yr) for yr in (2009, 2010, 2012)}
    prod2013_2016 = get_2013_2016_prod()
    
    pre_fl_prods_timeadjusted = [pd.Series(p.values, index=p.index + dt.timedelta(hours=1)) for p in pre_fl_prods.itervalues()]
    
    return pd.concat(pre_fl_prods_timeadjusted + [prod2013_2016])
    
#%%
def detect_outliers(prod, threshold):  
    surrounding_means = mean_of_hour_beforeandafter(prod)   
    roll_mean_1day = prod.rolling(window=24, center=True).mean()   
    outliers = prod[np.logical_and(np.abs(prod - surrounding_means) >= threshold,\
                                   np.abs(prod - roll_mean_1day) >= threshold)]
    
    return outliers


def clean_outliers_mean(prod, outliers):
    surrounding_means = mean_of_hour_beforeandafter(prod)
    cln_prod = prod.copy()
    for ix in outliers.index:
        cln_prod[ix] = surrounding_means[ix]
        
    return cln_prod

#%%

def get_2013_2016_prod():
    ts1_fl = dt.datetime(2013,1,1,1)
    ts2_fl = dt.datetime(2017,1,1,0)
    prod2013_2016 = sq.fetch_production(ts1_fl, ts2_fl)
    
    prod2013_2016_ser = pd.Series(prod2013_2016, index=pd.date_range(ts1_fl, ts2_fl, freq='H'))
    
    return prod2013_2016_ser


def peak_hour_hist(prod_series):
    max_hour = []
    for d in pd.date_range(prod_series.index[0], prod_series.index[-1], freq='D'):
        try:
            peak_time = prod_series[d:d+dt.timedelta(hours=23)].argmax()
            max_hour.append(peak_time.hour)      
        except:
            continue
    plt.figure()    
    plt.hist(max_hour, bins=np.arange(25)-0.5)
    

#%%

def excel_to_series(year=2009):
    df = pd.read_excel('data/raw_input/Varmeproduktion 2009 2015.xlsx', sheetname=str(year), header=2)
    
    return pd.Series(df.values.flatten(), index=df.index)


def mean_of_hour_beforeandafter(series):
    shifted_df = pd.DataFrame(index=series.index)
    shifted_df['prod-1h'] = series.shift(-1)
    shifted_df['prod1h'] = series.shift(1)
    surrounding_means = shifted_df.mean(axis=1)
    
    return surrounding_means
    
if __name__=="__main__":
    main(sys.argv[0])