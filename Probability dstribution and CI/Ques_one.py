# -*- coding: utf-8 -*-
"""
Calculate probability from the given dataset for the below cases

Data_set: Cars.csv
Calculate the probability of MPG of Cars for the below cases.
MPG <- Cars$MPG
a.	P(MPG>38)
b.	P(MPG<40)
c.	P (20<MPG<50)


@author: Radhakrishna Naik
"""
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

cars=pd.read_csv("c:/360DG/Datasets/Cars.csv")
sns.boxplot(cars.MPG)
#To calculate MPG >38
cars.MPG.describe()
x1=stats.norm.cdf(0.38,cars.MPG.mean(),cars.MPG.std())
x1
#To calculate p(MPG<40)
#max is 53.70-z value at 38
print("P(MPG>38 is  ",(53.70-x1))
########################################
#To calculate MPG<40
#first we will calculate probabilty at 40
#mean is 43.42 and std is 9.13
x2=stats.norm.cdf(0.40,34.42,9.13)
x2
#To calculate MPG less than 40 will be the same same
#0.72
#To calculate MPG lies between 20 and 50
x3=stats.norm.cdf(0.50,34.42,9.13)
x4=stats.norm.cdf(0.20,34.42,9.13)
x=x3-x4
x
