# -*- coding: utf-8 -*-
"""
Q2) Check whether the data follows normal distribution
a)	Check whether the MPG of Cars follows Normal Distribution Dataset: Cars.csv
b)	Check Whether the Adipose Tissue (AT) and Waist Circumference (Waist) from wc-at data set follows Normal Distribution
Dataset: wc-at.csv

@author: Radhakrishna Naik
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot 
cars=pd.read_csv("c:/360DG/Datasets/Cars.csv")
at=pd.read_csv("c:/360DG/Datasets/wc-at.csv")
sns.distplot(cars.MPG)
##QQ plot
from scipy import stats
import pylab
stats.probplot(cars["MPG"],dist="norm",plot=pylab)
#Data is normally distributed
sns.distplot(at.AT)
#Data is normally distributed
stats.probplot(at["AT"],dist="norm",plot=pylab)
#Majority points are on red line hence data is normally distributed
sns.distplot(at.Waist)
#Data is normally distributed
stats.probplot(at["Waist"],dist="norm",plot=pylab)
#majority points are near to red line