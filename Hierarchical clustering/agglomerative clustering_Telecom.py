# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:39:09 2022

@author: Dell
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
import matplotlib.pylab as plt
# Now import file from data set and create a dataframe
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
# Now apply this normalization function to crime datframe for all the rows and column from 1 until end
    
df_norm=norm_func(telecom.iloc[:,:])
# you can check the df_norm dataframe which is scaled between values from 0 to1
# you can apply describe function to new data frame
df_norm.describe()

# Now to create dendrogram, we need to measure distance,we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));plt.title("Hierarchical Clustering dendrogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

# applying agglomerative clustering choosing 11 as clusters from dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=11,linkage='complete',affinity="euclidean").fit(df_norm)
# apply labels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#Assign this series to Univ Dataframe as column and name the column as "clust"
telecom['clust']=cluster_labels
# we want to relocate the column 66 to 0 th position
telecom_new=telecom.iloc[:,[51,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]]
#now check the telecom_new dataframe
telecom_new.iloc[:,:].groupby(telecom_new.clust).mean()
telecom_new.iloc[:,:].groupby(telecom_new.clust).std()
telecom_new.to_csv("telecom.csv",encoding="utf-8")
import os
os.getcwd()
