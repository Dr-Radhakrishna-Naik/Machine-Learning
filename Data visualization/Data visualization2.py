# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:38:27 2022

@author: Dell
"""

import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
chick=pd.read_csv("c:/360DG/Datasets/ChickWeight.csv")
chick.describe()
# Weight###
# mean=121.818339
#Std.deviation=71.07
#median=103
#min=35 and Q1=63 ,Q1-min=63-35=28
#max=373 and Q3=163 max-Q3=373-163=210
# means weight is right skewed
plt.hist(chick.weight)
# Wight is right skewed
plt.boxplot(chick.weight)
# There are several outliers