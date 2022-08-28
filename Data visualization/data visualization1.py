# -*- coding: utf-8 -*-
"""
Calculate Skewness, Kurtosis using R/Python code & draw inferences on the following data.

@author: Dell
"""
from scipy.stats import skew
from scipy.stats import kurtosis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
cars=pd.read_csv("C:/360DG/Datasets/Statistical Datasets/Q1_a.csv")
cars.describe()
plt.hist(cars.speed)
#Data is normally distributed
plt.boxplot(cars.speed)
#There are no outliers in speed column
plt.hist(cars.dist)
#Dist.data is right skewed
plt.boxplot(cars.dist)
#There is one outliers in the data
#Kkeweness of speed
speed=cars['speed'].tolist()
speed
print("skewness of speed",skew(speed))
dist=cars['dist'].tolist()
print("skewness of dist",skew(dist))
print(skew(dist, axis=0, bias=True))
#it signifies that distribution is positively skewed
print(kurtosis(dist,axis=0,bias=True))
# it is playkurtic it means it produce less or extreme outliers
#############################
cars1=pd.read_csv("c:/360DG/Datasets/statistical Datasets/Q2_b.csv")
cars1.describe()
# mean of sp 121.54 and mean of WT 32.41
#Std deviation of SP is 14.18 and WT is 7.49
#median is SP=118.20 and WT=32.73
# min =99.56 Q1 of SP =113 and Q3=126 max is 169 min
#Q1-min is around 13 and Max-Q3 is 43 hence data SP is right skewed

# mean of WT 32.41
#Std deviation of  WT is 7.49
#median is  WT=32.73
# min =15.71 Q1 of SP =29.59 and Q3=37.39 max is 52.99
#Q1-min is around 14 and Max-Q3 is 14 hence data WT is normally distributed
plt.hist(cars1.SP)
plt.hist(cars1.WT) 
# to calculate skewness and kurtosis
sp=cars1['SP'].tolist()
print(skew(sp,axis=0,bias=True))
#1.5814536794423764 hence is right skewed
print(kurtosis(sp,axis=0,bias=True))
#2.7235214865269244 it is less than 3 ,SP is playkurtosis

wt=cars1["WT"].tolist()
print(skew(wt,axis=0,bias=True))
#-0.6033099322115126 it is slight left skewed
print(kurtosis(wt,axis=0,bias=True))
#0.8194658792266849<3 ,it is playkurtic

plt.boxplot(cars1.SP)
# There are outliers in the data
plt.boxplot(cars1.WT)
#There are outliers in WT