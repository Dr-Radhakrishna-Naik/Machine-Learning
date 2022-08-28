# -*- coding: utf-8 -*-
"""
A Government company claims that an average light bulb lasts 
270 days. A researcher randomly selects 18 bulbs for testing.
 The sampled bulbs last an average of 260 days, with a standard deviation of 90 days. If the CEO's claim were true, what is the probability that 18 randomly selected bulbs would have an average life of no more than 260 days

@author: Radhakrishna Naik
"""
from scipy import stats
from scipy.stats import norm
import math
#Assume null hypothesis is H0=Average life of bulb>=260
#Alternate hypothesis is H1=ave life of bulb <260
#find t score of 260
# t score=(s_mean-pop_mean)/(sample_sd_dev/sqr(n))
#s_mean=260,pop_mean=270,std_dev=90,n=18
t=(260-270)/(90/math.sqrt(18))
t
#To find p(x>260)
p_value=1-stats.t.cdf(abs(-0.4714),df=17)
p_value
