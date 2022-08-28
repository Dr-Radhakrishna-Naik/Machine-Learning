# -*- coding: utf-8 -*-
"""
The time required for servicing transmissions is normally 
distributed with  = 45 minutes and  = 8 minutes. The service manager plans to have work begin on the transmission of a customer’s 
car 10 minutes after the car is dropped 
off and the customer is told that the car will be ready 
within 1 hour from drop-off. What is the probability that
 the service manager cannot meet his commitment?

@author: Radhakrishna Naik
"""
#Average time=45 minutes, std.dev=8
#The work begins after 10 minutes
#ave_time=55 minutes,x=1 hour i.60 minutes
#z_value=(x-mean)/std.dev
#probability that customer will get car in 60 minutes

z_value=(60-55)/8
z_value
#using z_table corresponding value is 0.7323
#probability that customer will not get car in 60 minutes
no_commit=1-0.7323
no_commit
