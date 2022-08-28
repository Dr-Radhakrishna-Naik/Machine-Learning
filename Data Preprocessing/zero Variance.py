# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:25:33 2022

@author: Dell
"""


############Zero variance##########
import pandas as pd
df1=pd.read_csv("c:/360DG/Datasets/Z_dataset.csv ")
df1.dtypes
df1.var()    # variance in numeric value columns
#### zero variance and near zero variance ######

# If the variance is low or close to zero, then a feature is approximately 
# constant and will not improve the performance of the model.
# In that case, it should be removed.