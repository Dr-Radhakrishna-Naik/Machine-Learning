# -*- coding: utf-8 -*-
"""
Perform clustering analysis on the telecom dataset. The data is a mixture of both categorical and numerical data. It consists of the number of customers who churn. Derive insights and get possible information on factors that may affect the churn decision.
 Refer to Telco_customer_churn.xlsx dataset.
 1. Business Problem 
1.1.	What is the business objective?


 There's an intense competition in major telecom operators.
 Telecom market is turning from a seller's to buyer's market 
 and business-centric to customer-centric.
 Market operation and services need a higher requirement. 
 This determines the relationship between customer and telecom operators
 to turn to segmentation and personalization. 
 Nowadays,Telecom business-to-customer segment of the method is
 still based on experience or simple classification based on statistical methods.
 To understand the customer's overall composition
1.To understand group characteristics of various valuable customers
2.To understand group characteristics of loss customers
3.To understand consumption characteristics of customers
4.To understand group characteristics of customers with different credit rating
 
 1.2.	Are there any constraints?
 Should be able to process the data of large quantities of data points (millions of clients to the firm) in a few minutes
Privacy of the data is key (personal information)
Quality of results should be high, as mischarging a client can have high costs (especially at a large scale)
@author: Radhakrishna Naik

"""

#Telco customer churn
#Data Description:
#This sample data module tracks a fictional 
#telco company's customer churn based on various factors.
#The churn column indicates whether the customer departed within the last month.
# Other columns include gender, dependents, monthly charges, and 
#many with information about the types of services each customer has.
#The data set includes information about:
#Customers who left within the last month — the column is called Churn.
#Services that each customer has signed up for — phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
#Customer account information — how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
#Demographic info about customers — gender, age range, and if they have partners and dependents
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

telecom1=pd.read_excel("c:/360DG/Datasets/Telco_customer_churn.xlsx")
#EDA
telecom1.dtypes

# As follwing columns are  going to contribute hence drop it
telecom=telecom1.drop(["Customer ID","Count","Quarter","Referred a Friend","Offer","Contract","Payment Method"],axis=1)
telecom.describe()
# show the distribution of tenure.


plt.hist(data = telecom, x = 'Tenure in Months');
#This is apparently not a normal distribution.
# And with two peaks, there are two extreme kinds of people among all customers,
# and I will investigate what services have kept those who stay more than 70 months the most.
telecom1.Contract.value_counts()
#Month-to-Month    3610
#Two Year          1883
#One Year          1550
# Month to month subscribers  are more they are almost 2.5 times year to year subscribesr and 2-yrs to 2 yrs subscribers

#Let’s look at the phone service with multiple lines:
plt.hist(data = telecom, x = 'Monthly Charge');
# Aparantly normal distribution and 30 % customers higher monthly charge
plt.hist(data = telecom, x = 'Phone Service');
#phone service saying yes are more than No
plt.hist(data = telecom, x = 'Total Extra Data Charges ');
# There are several columns having ctegorical data,so create dummies for these
  #for all these columns create dummy variables
Num_Of_Referral_dummies=pd.DataFrame(pd.get_dummies(telecom['Number of Referrals']))
Phone_Service_dummies=pd.DataFrame(pd.get_dummies(telecom['Phone Service']))
Multiple_Lines_dummies=pd.DataFrame(pd.get_dummies(telecom['Multiple Lines']))
Internet_Service_dummies=pd.DataFrame(pd.get_dummies(telecom['Internet Service']))
Internet_Type_dummies=pd.DataFrame(pd.get_dummies(telecom['Internet Type']))
Online_Security_dummies=pd.DataFrame(pd.get_dummies(telecom['Online Security']))
Online_Backup_dummies=pd.DataFrame(pd.get_dummies(telecom['Online Backup']))
Device_Protection_Plan_dummies=pd.DataFrame(pd.get_dummies(telecom['Device Protection Plan']))
Premium_Tech_Support_dummies=pd.DataFrame(pd.get_dummies(telecom['Premium Tech Support']))
Streaming_TV_dummies=pd.DataFrame(pd.get_dummies(telecom['Streaming TV']))
Streaming_Movies_dummies=pd.DataFrame(pd.get_dummies(telecom['Streaming Movies']))
Streaming_Music_dummies=pd.DataFrame(pd.get_dummies(telecom['Streaming Music']))
Unlimited_Data_dummies=pd.DataFrame(pd.get_dummies(telecom['Unlimited Data']))
Paperless_Billing_dummies=pd.DataFrame(pd.get_dummies(telecom['Paperless Billing']))

## now let us concatenate these dummy values to dataframe
telecom=pd.concat([telecom,Num_Of_Referral_dummies,Phone_Service_dummies,Phone_Service_dummies,Multiple_Lines_dummies],axis=1)
telecom=pd.concat([telecom,Internet_Service_dummies,Internet_Type_dummies,Online_Security_dummies,Online_Backup_dummies],axis=1)

telecom=pd.concat([telecom,Device_Protection_Plan_dummies,Premium_Tech_Support_dummies,Streaming_TV_dummies,Streaming_Movies_dummies,
],axis=1)
telecom=pd.concat([telecom,Streaming_Music_dummies,Unlimited_Data_dummies,Paperless_Billing_dummies
],axis=1)
telecom=telecom.drop(["Number of Referrals","Phone Service","Multiple Lines","Internet Service","Internet Type","Online Security","Online Backup","Device Protection Plan","Premium Tech Support","Streaming TV","Streaming Movies","Streaming Music","Unlimited Data","Paperless Billing"],axis=1)
# we know that there is scale difference among the columns,which we have to remove
#either by using normalization or standardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
###########EDA

# Now apply this normalization function to airlines datframe for all the rows and column from 1 until end

df_norm=norm_func(telecom.iloc[:,:])
TWSS=[]
k=list(range(2,14))
# The values generated by TWSS are 12 and two get x and y values 12 by 12 ,range has been changed 2:14

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
TWSS

plt.plot(k,TWSS,'ro-');plt.xlabel("No_of_clusters");plt.ylabel("Total_within_SS")
# from the plot it is clear that the TWSS is reducing from k=2 to 3 and 3 to 4 
#than any other change in values of k,hence k=3 is selected
model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
telecom['clust']=mb
telecom.head()
telecom=telecom.iloc[:,[51,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]]

telecom.iloc[:,:].groupby(telecom.clust).mean()

telecom.to_csv("kmeans_telecom.csv",encoding="utf-8")
import os
os.getcwd()
