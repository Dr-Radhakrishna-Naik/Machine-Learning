# -*- coding: utf-8 -*-
"""
Let X ~ N(100, 20^2) its (100, 20 square).
Find two values, a and b, symmetric about the mean,
 such that the probability of the random variable taking 
 a value between them is 0.99.

@author: Dell
"""
import scipy.stats
#Given mean=100, and std_dev=20
#CI is 99 % ,lower limit=0.005 and upper is 0.995
#z value for 0.05 is
# we have equation for z
#z=(x-100)/20
#x=20Z+100
z_value=stats.norm.ppf(0.005)
z_value
x=20*z_value+100
x
#48.4834

#Z value at 0.995
z_value=stats.norm.ppf(0.995)
z_value
x=20*z_value+100
x
#151.51
# Two values are 48.5 and 151.6
