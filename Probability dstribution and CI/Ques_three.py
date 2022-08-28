# -*- coding: utf-8 -*-
"""
Q3) Calculate the Z scores of 90% confidence interval,
94% confidence interval, 60% confidence interval

@author: Radhakrishna Naik
"""
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
#To calculate z-score of 90% confidence interval
stats.norm.ppf(0.95)
stats.norm.ppf(0.05)
###########
#To calculate x-score of 94%
#0.03 and 
stats.norm.ppf(0.97)
stats.norm.ppf(0.03)
#To calculate z-score for 60% CI
stats.norm.ppf(0.80)
stats.norm.ppf(0.20)
