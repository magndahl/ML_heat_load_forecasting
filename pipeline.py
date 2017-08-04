# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:49:27 2017

@author: azfv1n8
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from build_data_set import categorical_columns


def df_to_np_arr_dict(df):
    
    return {'array':df.as_matrix(), 'headers':list(df.columns), 'index':list(df.index)}


def categoricals_to_dummies(Xdf, categoricals_in_Xdf):
    
    Xdf_withdummies = pd.get_dummies(Xdf, columns=categoricals_in_Xdf, drop_first=True)
    
    return Xdf_withdummies
    

def get_dummy_columns(Xdf_withdummies, categorical_columns=categorical_columns):
    
    return [col for col in Xdf_withdummies for cat_col in categorical_columns if col.startswith(cat_col)]
    

def get_dummy_col_ix(Xdf_withdummies, dummy_columns):
    
    return [list(Xdf_withdummies.columns).index(col) for col in dummy_columns]


    
class StandardScalerIgnoreDummies(object):
    
    def __init__(self, dummy_col_ix, standard_scaler, center_dummies=False):
        self.dummy_col_ix = dummy_col_ix
        self.standard_scaler = standard_scaler
        self.center_dummies = center_dummies
        
    
    def transform(self, X):        
        return self.__partial_tranformation__(X, cont_transform_func=self.standard_scaler.fit_transform)
    
    
    def fit(self, X, y=None):
        return self        
    
    
    def inverse_transform(self, X):
        return self.__partial_tranformation__(X, cont_transform_func=self.standard_scaler.inverse_transform)
    
    
    def inverse_transform_y(self, y):
        """ This function assumes that transform has been called on an
            array with y as the first column!
            
            """
        scale = self.standard_scaler.scale_[0]
        mean = self.standard_scaler.mean_[0]
        
        return scale*y + mean
    

    def __partial_tranformation__(self, X, cont_transform_func):
        cont_col_ix = [i for i in range(X.shape[1]) if i not in self.dummy_col_ix]        
        cont_X = np.concatenate([X[:,i:i+1] for i in cont_col_ix], axis=1)
        cont_X_scaled = cont_transform_func(cont_X)
        
        X_res = np.empty_like(X)
        for i in range(X.shape[1]):
            if i in self.dummy_col_ix:
                if not self.center_dummies:
                    X_res[:,i] = X[:,i]
                elif self.center_dummies:
                    X_res[:,i] = 2*X[:,i] - 1 # This transformes [0, 1] to [-1, 1]
            elif i in cont_col_ix:
                ix_in_cont_X_scaled = cont_col_ix.index(i)
                X_res[:,i] = cont_X_scaled[:, ix_in_cont_X_scaled]
        
        return X_res
        
        
