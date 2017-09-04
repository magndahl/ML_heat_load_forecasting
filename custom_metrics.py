# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:16:05 2017

@author: azfv1n8
"""
import numpy as np
from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/y_true))


def mean_error(y_true, y_pred):
    return np.mean((y_pred - y_true))