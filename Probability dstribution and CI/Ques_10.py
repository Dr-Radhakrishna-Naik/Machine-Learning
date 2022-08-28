# -*- coding: utf-8 -*-
"""
Consider a company that has two different divisions.
 The annual profits from the two divisions are independent 
 and have distributions Profit1 ~ N(5, 3^2) and 
 Profit2 ~ N(7, 4^2) respectively. 
 Both the profits are in $ Million. 
 Answer the following questions about the total profit of the company in Rupees. Assume that $1 = Rs. 45
A.	Specify a Rupee range (centered on the mean) such that it contains 95% probability for the annual profit of the company.
B.	Specify the 5th percentile of profit (in Rupees) for the company
C.	Which of the two divisions has a larger probability of making a loss in a   given year?

@author: Radhakrishna Naik
"""
import numpy as np
from scipy import stats
from scipy.stats import norm
#Let us try to under the given entities
#mean profit of the company, adding two depts mean
mean=5+7
mean
print("Mean profit of the company is ",mean*45," Million Rupees")
#variations of profits from two different division is 3 and 4
#Std_dev of the company is
SD=np.sqrt(9+16)
SD
print("variation of profit of the company is ",SD*45,"Million Rupees")
#Given confidence interval is 95%,lower limit is 0.025 and upper
#limit is 0.975
# z_value for 0.025
z1=stats.norm.ppf(0.025)
z1
#z_value for 0.975
z2=stats.norm.ppf(0.975)
z2
#we have mean=12*45 =540and SD=5*45=225 Rupees
#z=x-mean/std_dev
#x=225z+540
#lower limit value of x
x1=225*z1+540
x1
#Upperlimit value of x
x2=225*z2+540
x2
print("Rupee range is ",x1,x2,"in Million")
###########################################
#To compute 5 th percentile of profit fo the company
#we will use x=z*std_dev+mean
#from z table 5 th percentile value is -1.645
x=540+(-1.645)*225
x
print("5 th percentile value in million rupees is  ",np.round(x))
############################
#Which of the two divisions has larger probability of making loss
#prob of div1 making loss p(x<0)
stats.norm.cdf(0,5,3)
#0.047
stats.norm.cdf(0,7,4)
#0.0400