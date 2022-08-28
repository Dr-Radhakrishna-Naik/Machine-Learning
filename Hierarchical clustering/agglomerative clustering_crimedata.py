# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:39:09 2022

@author: Dell
"""
#Perform Clustering for the crime data and identify the number of clusters formed and draw inferences.

#Data Description:
#Murder -- Muder rates in different places of United States
#Assualt- Assualt rate in different places of United States
#UrbanPop - urban population in different places of United States
#Rape - Rape rate in different places of United States



import pandas as pd
import matplotlib.pylab as plt
# Now import file from data set and create a dataframe
crime1=pd.read_csv("c:/360DG/Datasets/crime_data.csv")
#EDA
# As state column is not going to contribute hence drop it
crime=crime1.drop(["Unnamed: 0"],axis=1)
crime.describe()

# we know that there is scale difference among the columns,which we have to remove
#either by using normalization or standardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Now apply this normalization function to crime datframe for all the rows and column from 1 until end
    
df_norm=norm_func(crime.iloc[:,:])
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

# applying agglomerative clustering choosing 4 as clusters from dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity="euclidean").fit(df_norm)
# apply labels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#Assign this series to Univ Dataframe as column and name the column as "clust"
crime['clust']=cluster_labels
# we want to relocate the column 10 to 0 th position
crime_new=crime.iloc[:,[4,0,1,2,3]]
#now check the airlines dataframe
crime_new.iloc[:,:].groupby(crime_new.clust).mean()
crime_new.iloc[:,:].groupby(crime_new.clust).std()
crime_new.to_csv("crime.csv",encoding="utf-8")
import os
os.getcwd()
