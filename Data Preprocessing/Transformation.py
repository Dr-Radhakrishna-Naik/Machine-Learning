# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:01:23 2022

@author: Dell
"""
################Transformation#############
import pandas as pd
import scipy.stats as stats
import pylab
data=pd.read_csv("c:/360DG/Datasets/calories_consumed.csv ")
data.dtypes
stats.probplot(data.Weight_gain,dist="norm",plot=pylab)
stats.probplot(data.Cal,dist="norm",plot=pylab)
import numpy as np
### To carry on transformation
stats.probplot(np.log(data.Weight_gain),dist="norm",plot=pylab)
stats.probplot(np.log(data.Cal),dist="norm",plot=pylab)
