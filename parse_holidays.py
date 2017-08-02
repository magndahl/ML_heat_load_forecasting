# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:50:56 2017

@author: azfv1n8
"""

import pandas as pd
import datetime as dt
from collections import OrderedDict
import cPickle as pickle

savepath = 'data/cleaned/holidays/'
load_path = 'data/raw_input/'


def main():
    dfs_dict = load_all_holidays()
    observances_dates = listofdates_by_holidaytype(dfs_dict, holidaytype='Observance')
    save_dates_list(observances_dates, 'observances2009_2018.pkl')
    
    national_holidays_dates = listofdates_by_holidaytype(dfs_dict, holidaytype='National holiday')
    save_dates_list(national_holidays_dates, 'national_holidays2009_2018.pkl')
    
    school_holiday_dicts = load_all_school_holidays()
    school_holiday_datelist = listdates_school_holidays(school_holiday_dicts)
    save_dates_list(school_holiday_datelist, 'school_holidays2009_2017.pkl')
    
    return


def load_all_holidays():
    folder = 'Danish holidays/'
    dfs_dict = {}
    for yr in range(2009,2019):
        filename = 'Danish_holidays_' + str(yr)
        df = pd.read_csv(load_path + folder + filename, delimiter='\t')
        df_with_parsed_dates = parse_dates(df, yr)
        dfs_dict[yr] = df_with_parsed_dates

    return dfs_dict


def parse_dates(df, year):
    df['Datetime'] = [dt.datetime.strptime(str(year) + date_string, '%Y%b %d') for date_string in df['Date']]
    
    return df


def listofdates_by_holidaytype(dfs_dict, holidaytype='Observance'):
    date_list = []
    
    for df in dfs_dict.itervalues():
        date_list.extend(df[df['Holiday Type']==holidaytype]['Datetime'])
          
    return date_list


def save_dates_list(date_list, filename):
    
    with open(savepath + filename, 'w') as f:
        pickle.dump(date_list, f)
        
    return


def load_all_school_holidays():
    folder = 'School holidays/'
    filename = 'school_holidays.xlsx'
    dfs_dict = OrderedDict()
    for yr in range(2009, 2018):
        df = pd.read_excel(load_path+folder+filename, sheetname=str(yr))
        dfs_dict[yr] = df
        
    return dfs_dict
        

def listdates_school_holidays(dfs_dict):
    dates_list = []
    for yr, df in zip(dfs_dict.iterkeys(), dfs_dict.itervalues()):
        for vacation in df.columns:
            print yr, vacation
            first_day = df.ix[0, vacation]
            last_day = df.ix[-1, vacation]
            dates_list.extend(pd.date_range(first_day, last_day, freq='D'))

    return dates_list


if __name__=="__main__":
    main()