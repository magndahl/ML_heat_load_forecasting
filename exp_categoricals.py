# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:01:04 2017

@author: azfv1n8
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import datetime as dt

df = pd.DataFrame(index = pd.date_range(dt.datetime(2016,1,1,1), dt.datetime(2017,1,1,0), freq='H'))

df['y'] = np.random.randn(len(df.index))
df['week'] = [d.week for d in df.index]
df['weekend'] = [(d.dayofweek in (5,6)) for d in df.index]


le = LabelEncoder()
le.fit(df['week'])

le2 = LabelEncoder()
le2 = le.fit(df['weekend'])

df['weekend_enc'] = le2.transform(df['weekend'])


# this is better:
    
    
df_with_dummies = pd.get_dummies(df, columns=['week', 'weekend'])