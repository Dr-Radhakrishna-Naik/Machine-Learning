# -*- coding: utf-8 -*-
"""
Perform clustering analysis on the telecom dataset. The data is a mixture of both categorical and numerical data. It consists of the number of customers who churn. Derive insights and get possible information on factors that may affect the churn decision.
 Refer to Telco_customer_churn.xlsx dataset.
@author: Radhakrishna Naik
1. Business Problem 
1.1.	What is the business objective?
Customer segmentation is critical for auto insurance companies to gain competitive advantage by mining useful customer related information.
 While some efforts have been made for customer segmentation to support auto insurance decision making, 
 their customer segmentation results tend to be affected by the characteristics of the algorithm used and lack multiple validation from multiple algorithms.
1.2.	Are there any constraints?
there is mixed data,categorical and numerical data

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
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
# Now import file from data set and create a dataframe
autoi=pd.read_csv("c:/360DG/Datasets/AutoInsurance.csv")
#EDA
autoi.info
autoi.dtypes
autoi.describe()
#The average customer lifetime value is 8004 and min is 1898 and max is 83325

# As follwing columns are  going to contribute hence drop it
autoi1=autoi.drop(["Customer","State","Education","Sales Channel","Effective To Date"],axis=1)

plt.hist(data = autoi1, x = 'Customer Lifetime Value');
#This is apparently not a normal distribution.
# And with one peak indicate customer lifetime value of 100000 is higher
plt.hist(data = autoi1, x = 'Income');
#This is apparently not a normal distribution.lower income customers are more
plt.hist(data = autoi1, x = 'Monthly Premium Auto');
# lower premium customers are more


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

# Now apply this normalization function to airlines datframe for all the rows and column from 1 until end

df_norm=norm_func(autoi_new.iloc[:,:])
TWSS=[]
k=list(range(2,26))
# The values generated by TWSS are 24 and two get x and y values 24 by 24 ,range has been changed 2:26
#again restart the kernel and execute once
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
autoi_new['clust']=mb
autoi_new.head()
autoi_new=autoi_new.iloc[:,[51,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]]

autoi_new.iloc[:,:].groupby(autoi_new.clust).mean()

autoi_new.to_csv("kmeans_autoi_new.csv",encoding="utf-8")
import os
os.getcwd()

