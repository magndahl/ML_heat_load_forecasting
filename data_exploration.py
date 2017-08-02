# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:28:42 2017

@author: azfv1n8
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pipeline as pl
from build_data_set import categorical_columns



df = pd.read_pickle('data/cleaned/assembled_data/cv_data.pkl')

df_w_dummies = pl.categoricals_to_dummies(df, categorical_columns)


arr_dict = pl.df_to_np_arr_dict(df_w_dummies)

dummy_columns = pl.get_dummy_columns(df_w_dummies, categorical_columns)
dummy_column_ix = pl.get_dummy_col_ix(df_w_dummies, dummy_columns)

scaler = pl.StandardScalerIgnoreDummies(dummy_column_ix, StandardScaler())


arr = arr_dict['array']

arr_scaled = scaler.transform(arr)


def all_joint_plots(arr_scaled, variables=arr_dict['headers']):
    
    scaled_df = pd.DataFrame(data=arr_scaled, columns=variables)
    figpath = 'figures/data_exploration/'
    
    for i, pred_var in enumerate(variables[1:]):
        filename = 'prod_vs_' + pred_var + '.png'
        sns.jointplot(x=pred_var, y='prod', data=scaled_df, alpha=0.1)
        plt.savefig(figpath + filename)
        plt.close()
    
    return


def corr_coeff_plot(arr_scaled, variables=arr_dict['headers']):
    
    corr_mat = np.corrcoef(arr_scaled.transpose())
    plt.pcolormesh(corr_mat, cmap=plt.cm.get_cmap('viridis'))
    plt.colorbar()
    
    return