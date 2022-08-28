# -*- coding: utf-8 -*-
"""
The current age (in years) of 400 clerical employees 
at an insurance claims processing center is normally distributed
 with mean  = 38 and Standard deviation =6.
 For each statement below, please specify True/False. 
 If false, briefly explain why.
A.	More employees at the processing center are older than 44 than between 38 and 44.
B.	A training program for employees under the age of 30 at the center would be expected to attract about 36 employees.


@author: Radhakrishna Naik
"""
#mean=38, std_dev=6 ,x=44
z_value=(44-38)/6
z_value
#corrsponding value for z=1 is 84.13%
#There are 84.13 people of age 44 or less
#% of people having age more than 44
more_than_44=100-84.13
more_than_44
#out of 100,15.87 people are having age 44 above
#out of 400 63 people are having age more than 44
more_than_44=4*15.87
more_than_44
#63
#Now let us calculate z at 38
z_value=(38-38)/6
z_value
#corresponding value from z-table is 50%
# % of people having age 44 is 84.13 and 38 is 50%
#people between 38 and 44 age=84.13-50=34.13
#Total 137 out 400 people are in this age group
#They are more than people having age more_than_44
###################################################
##b  A training program under 30 
z_value=(30-38)/6
z_value
# -1.3333333, coresponding probability value is 9.15%
# out of 100,9.15 people are getting training
#out of 400 it is 36
#hence true