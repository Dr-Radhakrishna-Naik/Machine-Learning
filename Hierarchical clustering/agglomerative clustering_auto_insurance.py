# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:39:09 2022

@author: Dell
"""
#Data Description
#1. Customer - Customer ID, it is unique value
#2. State - There are five location where customers live in states (Washington,Arizona, Nevada, California, Oregon)
#3. Customer Lifetime Value - Value of customers insurance
#4. Response - This will be our dependent variable. with categorical response “Yes” if the customers would like to renew their insurance and “No” if the customers would discontinue their insurance.
#5. Coverage - There are 3 types of coverage insurances (Basic, Extended and Premium)
#6. Education - Background education of customers (High School or Below, Bachelor, College, Master and Doctor)
#7. Effective To Date - The first date when customer would like to actived their car insurance
#8. Employment Status - Customer employemen status whether they are Employed, Unemployed, Medical Leave, Disabled, or Retired
#9. Gender - F for Female and M for Male
#10. Income - Customers income
#11. Location Code - Where the customers live likes in Rural, Suburban, and Urban.
#12. Marital Status - Customer marital status (Divorced, Married or Single)
#13. Monthly Premium Auto - Premium auto that customers need to pay every month
#14. Months Since Last Claim - Number of months since customers did last claim
#15. Months Since Policy Inception - Number of months since customers did policy inception
#16. Number of Open Complaints - Number of complaints
#17. Number of Policies - Number of policies in when customers take part of car insurance
#18. Policy Type - There are three type of policies in car insurance (Corporate Auto, Personal Auto, and Special Auto)
#19. Policy - 3 variety of policies in insurance. There are three policies in each policy types (Corporate L3, Corporate L2, Corporate L1, Personal L3,Personal L2, Personal L1,Special L3, Special L2, Special L1)
#20. Renew Offer Type - Each sales of Car Insurance offer 4 type of new insurances to customers. There are Offer 1, Offer 2, Offer 3 and Offer 4
#21. Sales Channel - Each sales offer new car insurance by Agent, Call Center, Web and Branch
#22. Total Claim Amount - Number of Total Claim Amount when customer did based on their coverage and other considerations.
#23. Vehicle Class - Type of vehicle classes that customers have Two-Door Car, Four-Door Car SUV, Luxury SUV, Sports Car, and Luxury Car
#24. Vehicle Size - Type of customers vehicle size, there are small, medium and large

import pandas as pd
import matplotlib.pylab as plt
# Now import file from data set and create a dataframe
autoi=pd.read_csv("c:/360DG/Datasets/AutoInsurance.csv")
#EDA
autoi.info
autoi.dtypes
# As follwing columns are  going to contribute hence drop it
autoi1=autoi.drop(["Customer","State","Education","Sales Channel","Effective To Date"],axis=1)
autoi1.describe()
# show the distribution of tenure.
plt.hist(data = autoi1, x = 'Customer Lifetime Value');
#This is apparently not a normal distribution.
# And with one peak indicate customer lifetime value of 100000 is higher
plt.hist(data = autoi1, x = 'Income');
#This is apparently not a normal distribution.lower income customers are more
plt.hist(data = autoi1, x = 'Monthly Premium Auto');
#There are lower premium customers


# There are several columns having ctegorical data,so create dummies for these
  #for all these columns create dummy variables
Response_dummies=pd.DataFrame(pd.get_dummies(autoi1['Response']))
Coverage_dummies=pd.DataFrame(pd.get_dummies(autoi1['Coverage']))
Employment_Status_dummies=pd.DataFrame(pd.get_dummies(autoi1['EmploymentStatus']))
Gender_dummies=pd.DataFrame(pd.get_dummies(autoi1['Gender']))
LocationCode_dummies=pd.DataFrame(pd.get_dummies(autoi1['Location Code']))
Marital_Status_dummies=pd.DataFrame(pd.get_dummies(autoi1['Marital Status']))
Policy_Type_dummies=pd.DataFrame(pd.get_dummies(autoi1['Policy Type']))
Policy_dummies=pd.DataFrame(pd.get_dummies(autoi1['Policy']))
Renew_Offer_Type_dummies=pd.DataFrame(pd.get_dummies(autoi1['Renew Offer Type']))
Vehicle_Class_dummies=pd.DataFrame(pd.get_dummies(autoi1['Vehicle Class']))
Vehicle_Size_dummies=pd.DataFrame(pd.get_dummies(autoi1['Vehicle Size']))


## now let us concatenate these dummy values to dataframe
autoi_new=pd.concat([autoi1,Response_dummies,Coverage_dummies,Employment_Status_dummies,Gender_dummies,LocationCode_dummies,Marital_Status_dummies,Policy_Type_dummies,Policy_dummies,Renew_Offer_Type_dummies,Vehicle_Class_dummies,Vehicle_Size_dummies],axis=1)

autoi_new=autoi_new.drop(["Response","Coverage","EmploymentStatus","Gender","Location Code","Marital Status","Policy Type","Policy","Renew Offer Type","Vehicle Class","Vehicle Size"],axis=1)
# we know that there is scale difference among the columns,which we have to remove
#either by using normalization or standardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Now apply this normalization function to crime datframe for all the rows and column from 1 until end
    
df_norm=norm_func(autoi_new.iloc[:,:])
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
h_complete=AgglomerativeClustering(n_clusters=21,linkage='complete',affinity="euclidean").fit(df_norm)
# apply labels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#Assign this series to Univ Dataframe as column and name the column as "clust"
autoi_new['clust']=cluster_labels
# we want to relocate the column 66 to 0 th position
autoi_new1=autoi_new.iloc[:,[51,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]]
#now check the telecom_new dataframe
autoi_new1.iloc[:,:].groupby(autoi_new1.clust).mean()
autoi_new1.iloc[:,:].groupby(autoi_new1.clust).std()
autoi_new1.to_csv("autoi_new1",encoding="utf-8")
import os
os.getcwd()
