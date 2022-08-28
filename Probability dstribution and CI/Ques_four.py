# -*- coding: utf-8 -*-
"""
Q4) Calculate the t scores of 95% confidence interval, 96% confidence interval, 99% confidence interval 
for sample size of 25

@author: Radhakrishna Naik
"""
from scipy import stats
from scipy.stats import norm

#t score of 95% CI of sample size 25
#df=n-1=25-1=24
stats.t.ppf(0.975,24)
stats.t.ppf(0.025,24)
######
#t score of 96% CI of sample size 25
#df=n-1=25-1=24
stats.t.ppf(0.98,24)
stats.t.ppf(0.02,24)
###############
#t score of 99% CI of sample size 25
#df=n-1=25-1=24
stats.t.ppf(0.995,24)
stats.t.ppf(0.005,24)
