# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:39:09 2022

@author: Dell
"""

#The data used for this analysis contains information on 4,000 passengers who belong to an airline’s frequent flier program. For each passenger, the data include information on their mileage history and 
#on different ways they accrued or spent miles in the last year.


#Data Description
#1. ID - Customer ID, it is unique value 
#2. Balance - Number of miles eligible for award travel
#3. Qual_miles -Number of miles counted as qualifying for Topflight status
#4. cc1_miles: Number of miles earned with freq.flyer credit card in past 12 Months
#5. cc2_miles months:Number of miles earned with freq.flyer credit card in past 12 Months

#6. cc3_miles - Number of miles earned with freq.flyer credit card in past 12 Months
#7. Bonus miles - The miles earned for non-flight bonus transaction in past 12 Months
#8. Bonus trans - Number of non-flight bonus transactions in past 12 months
#9. Flight_miles_12mo -number of flight miles in past 12 months
#10.Flight_trans_12 - Number of flight transactions in the past 12 Months
#11. Days since enroll - Number of days since enrolled date.
#12. Award?- Dummy variable for last award)
#
import pandas as pd
import matplotlib.pylab as plt
# Now import file from data set and create a dataframe
airlines=pd.read_excel("c:/360DG/Datasets/EastWestAirlines.xlsx")
#EDA
airlines.info
airlines.dtypes
# As follwing columns are  going to contribute hence drop it
airlines1=airlines.drop(["ID#","Award?"],axis=1)
airlines1.describe()
# show the distribution of tenure.
plt.hist(data = airlines1, x = 'Balance');
#This is apparently not a normal distribution.
# And with one peak indicate customer is eligible for award travel is between 0 to 0.25
plt.hist(data = airlines1, x = 'Qual_miles');
#The most of cutomers are eligible for 1000 miles travel in top flights
plt.hist(data = airlines1, x = 'cc1_miles');
#Majority customers are eligible for 1 to 1.5 miles in a year

plt.hist(data = airlines1, x = 'Bonus_miles');
#Majority customers are eligible for bonus of  1 to 50000 miles



# we know that there is scale difference among the columns,which we have to remove
#either by using normalization or standardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Now apply this normalization function to crime datframe for all the rows and column from 1 until end
    
df_norm=norm_func(airlines1.iloc[:,:])
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
airlines1['clust']=cluster_labels
# we want to relocate the column 66 to 0 th position
airlines1=airlines1.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]
#now check the telecom_new dataframe
airlines1.iloc[:,:].groupby(airlines1.clust).mean()
airlines1.iloc[:,:].groupby(airlines1.clust).std()
airlines1.to_csv("airlines",encoding="utf-8")
import os
os.getcwd()
