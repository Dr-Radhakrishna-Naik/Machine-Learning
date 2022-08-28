# -*- coding: utf-8 -*-
"""
Suppose we want to estimate the average weight of an adult male
 in Mexico. We draw a random sample of 2,000 men from 
 a population of 3,000,000 men and weigh them. 
 We find that the average person in our sample weighs
 200 pounds, and the
 
standard deviation of the sample is 30 pounds. Calculate 94%,98%,96% confidence interval?


@author: Dell
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
# Avg. weight of Adult in Mexico with 94% CI
stats.norm.interval(0.94,200,30/(2000**0.5))
# Avg. weight of Adult in Mexico with 98% CI
stats.norm.interval(0.98,200,30/(2000**0.5))
# Avg. weight of Adult in Mexico with 96% CI
stats.norm.interval(0.96,200,30/(2000**0.5))