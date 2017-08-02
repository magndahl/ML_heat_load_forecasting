# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:31:02 2017

@author: azfv1n8
"""

import numpy as np
import pandas as pd

absolute_zero = -273.15 # degree Celcius

load_path_tmp = 'data/raw_input/RE_atlas_data/Aarhus_tmp2m_1979_2010/'
load_path_tmp_new = 'data/raw_input/RE_atlas_data/Aarhus_tmp2m_2015_2016/'
load_path_wind_solar = 'data/raw_input/RE_atlas_data/Aarhus_wind_solar_1979_2016/'

load_path_forecast = 'data/raw_input/DMI ens forecasts/'

lats_tmp = np.load(load_path_tmp + 'latitudes.npy')
longs_tmp = np.load(load_path_tmp + 'longitudes.npy')
dates_tmp = np.load(load_path_tmp + 'dates.npy')
temps = np.load(load_path_tmp + 'temperature.npy')

lats_new = np.load(load_path_tmp_new + 'latitudes.npy')
longs_new = np.load(load_path_tmp_new + 'longitudes.npy')
dates_new = np.load(load_path_tmp_new + 'dates.npy')
temps_new = np.load(load_path_tmp_new + 'temperature.npy')

lats_wind_solar = np.load(load_path_wind_solar + 'latitudes.npy')
longs_wind_solar = np.load(load_path_wind_solar + 'longitudes.npy')
dates_wind_solar = np.load(load_path_wind_solar + 'dates.npy')

solar_influx = np.load(load_path_wind_solar + 'influx.npy') # this is solar irradiance
solar_outflux = np.load(load_path_wind_solar + 'outflux.npy') # this will be ignored

wind_speed = np.load(load_path_wind_solar + 'wind_speed.npy')

# use this to check that time zone is consistent with atlas data. Same convention, local time is used
sunRad_forecast = np.load(load_path_forecast + 'sunRad_geo71699_2016010101_to_2017010100.npy') 

def main():
    lats_ix = 0
    longs_ix = 0
    
    assert(all((lats_tmp==lats_new).flatten()))
    assert(all((lats_tmp==lats_wind_solar).flatten()))
    assert(all((longs_tmp==longs_new).flatten()))
    assert(all((longs_tmp==longs_wind_solar).flatten()))
    
    print "Location of point for weather data", lats_tmp[lats_ix, longs_ix], longs_tmp[lats_ix, longs_ix]
    
    
    time_steps = pd.date_range(pd.datetime.fromtimestamp(dates_wind_solar[0]), pd.datetime.fromtimestamp(dates_wind_solar[-1]), freq='H')
    df = pd.DataFrame(index=time_steps, columns=['Tout', 'vWind', 'sunRad'], dtype=float)
    insert_Tout_celcius(df, lats_ix, longs_ix)
    insert_wind_and_solar(df, lats_ix, longs_ix)
    
    # handle missing values (fill forward) due to daylight savings time
    df = df.fillna(method='ffill')
    
    # correct negative solar values, set to 0
    mask = df.sunRad < 0
    df.loc[mask, 'sunRad'] = 0

    save=False
    if save:
        df.to_pickle('data/cleaned/RE_atlas_weather_1979_2016.pkl')
        
    return df



def insert_Tout_celcius(df, lats_ix, longs_ix):
    for t, T in zip(dates_tmp, temps[:,lats_ix, longs_ix] + absolute_zero):
        dt_timestep = pd.datetime.fromtimestamp(t)
        df.at[dt_timestep, 'Tout'] = T
        
    for t, T in zip(dates_new, temps_new[:,lats_ix, longs_ix] + absolute_zero):
        dt_timestep = pd.datetime.fromtimestamp(t)
        df.at[dt_timestep, 'Tout'] = T

    return df


def insert_wind_and_solar(df, lats_ix, longs_ix):
    for t, w, s in zip(dates_wind_solar, wind_speed[:,lats_ix, longs_ix], solar_influx[:,lats_ix, longs_ix]):
        dt_timestep = pd.datetime.fromtimestamp(t)
        df.at[dt_timestep, 'vWind'] = w
        df.at[dt_timestep, 'sunRad'] = s
        
    return df


if __name__=="__main__":
    main()