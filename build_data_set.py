# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:48:03 2017

@author: azfv1n8
"""

import pandas as pd
from collections import OrderedDict
import cPickle as pickle

load_path = 'data/cleaned/'


def main(save=False):
    df_full = assemble_dataset_no_dummies()
    
    if save:
        save_cv_and_test_data(df_full)
        
    return
    

def assemble_dataset_no_dummies():
    prod = all_col_retrievers['prod']()
    df = pd.DataFrame(index=prod.index)
    
    for col_name in all_col_retrievers.iterkeys():
        print col_name
        column = all_col_retrievers[col_name](df)
        df[col_name] = column
    
    return df
             
    
def load_cleaned_prod_col(*args):    
    prod = pd.read_pickle(load_path + 'production2009to2016_not2011.pkl')
    
    return prod


def prod_lag1_col(df):
    return lag_prod(df['prod'], 1)


def prod_lag24_col(df):
    return lag_prod(df['prod'], 24)


def prod_lag48_col(df):
    return lag_prod(df['prod'], 48)


def prod_lag24or48_col(df):
    lagged24 = prod_lag24_col(df)
    lagged48 = prod_lag48_col(df)
    
    last_avail_hour_at_pred_time_left = 7 # it is necessary to use the left stamped convention
                                         # to handle hour 0 being the last hour in the day
                                         # in the right stamped convvention, which we use everywhere else
    prod_24or48_lags = [lagged24[ts] if timesteps_left_stamped(ts).hour<=last_avail_hour_at_pred_time_left else lagged48[ts] \
                        for ts in df.index]
    
    return prod_24or48_lags


def prod_lag168_col(df):
    return lag_prod(df['prod'], 168)


def load_Tout_col(df):
    atlas_weather = pd.read_pickle(load_path + 'RE_atlas_weather_1979_2016.pkl')
    
    return atlas_weather.ix[df.index, 'Tout']


def load_vWind_col(df):
    atlas_weather = pd.read_pickle(load_path + 'RE_atlas_weather_1979_2016.pkl')
    
    return atlas_weather.ix[df.index, 'vWind']


def load_sunRad_col(df):
    atlas_weather = pd.read_pickle(load_path + 'RE_atlas_weather_1979_2016.pkl')
    
    return atlas_weather.ix[df.index, 'sunRad']


def is_weekend_col(df):
    left_time_steps = timesteps_left_stamped(df.index)
    
    return [(ts.weekday_name in ('Saturday', 'Sunday')) for ts in left_time_steps]


def is_observance_col(df):
    with open(load_path + 'holidays/observances2009_2018.pkl', 'r') as f:
        observance_date_list = pickle.load(f)      
    left_time_steps = timesteps_left_stamped(df.index)
    
    return [(round_down_to_date(ts) in observance_date_list) for ts in left_time_steps]


def is_national_holiday_col(df):
    with open(load_path + 'holidays/national_holidays2009_2018.pkl', 'r') as f:
        national_holiday_date_list = pickle.load(f)
    left_time_steps = timesteps_left_stamped(df.index)
    
    return [(round_down_to_date(ts) in national_holiday_date_list) for ts in left_time_steps]


def is_school_holiday_col(df):
    with open(load_path + 'holidays/school_holidays2009_2017.pkl', 'r') as f:
        school_holiday_date_list = pickle.load(f)       
    left_time_steps = timesteps_left_stamped(df.index)
        
    return [(round_down_to_date(ts) in school_holiday_date_list) for ts in left_time_steps]


def hour_col(df):
    return [ts.hour for ts in df.index]


def weekday_col(df):
    left_time_steps = timesteps_left_stamped(df.index)
    
    return [ts.dayofweek for ts in left_time_steps]


def month_col(df):
    left_time_steps = timesteps_left_stamped(df.index)

    return [ts.month for ts in left_time_steps]


def lag_prod(prod, nlags):  
    return prod.shift(nlags).fillna(method='backfill')


def timesteps_left_stamped(timesteps):
    return timesteps + pd.Timedelta(hours=-1)

def round_down_to_date(pd_datetime):
    return pd.tslib.Timestamp(pd.datetime(pd_datetime.year, pd_datetime.month, pd_datetime.day, 0, 0))


def save_cv_and_test_data(full_df, split_date=pd.datetime(2016,1,1,0)):
    cv_df = full_df[full_df.index <= split_date]
    test_df = full_df[full_df.index > split_date]
    
    cv_df.to_pickle('data/cleaned/assembled_data/cv_data.pkl')
    test_df.to_pickle('data/cleaned/assembled_data/test_data.pkl')
    
    return cv_df, test_df
    

all_col_retrievers = OrderedDict([('prod', load_cleaned_prod_col),
                                  ('prod_lag1', prod_lag1_col),
                                  ('prod_lag24', prod_lag24_col),
                                  ('prod_lag48', prod_lag48_col),
                                  ('prod_lag168', prod_lag168_col),
                                  ('prod_lag24or48', prod_lag24or48_col),
                                  ('Tout', load_Tout_col),
                                  ('vWind', load_vWind_col),
                                  ('sunRad', load_sunRad_col),
                                  ('weekend', is_weekend_col),
                                  ('observance', is_observance_col),
                                  ('national_holiday', is_national_holiday_col),
                                  ('school_holiday', is_school_holiday_col),
                                  ('hour', hour_col),
                                  ('weekday', weekday_col),
                                  ('month', month_col)])

categorical_columns = ['weekend', 'observance', 'national_holiday', \
                       'school_holiday', 'hour', 'weekday', 'month']

if __name__=='__main__':
    main()