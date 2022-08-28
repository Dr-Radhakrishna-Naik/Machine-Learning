# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:51:29 2022

@author: Dell
"""

#############standardization
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
d=pd.read_csv("c:/360DG/Datasets/Seeds_data.csv")
a=d.describe()
a
scaler=StandardScaler()
df=scaler.fit_transform(d)
dataset=pd.DataFrame(df)


