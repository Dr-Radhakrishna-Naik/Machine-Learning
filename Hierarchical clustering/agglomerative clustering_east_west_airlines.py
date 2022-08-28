# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:39:09 2022

@author: Dell
"""


import pandas as pd
import matplotlib.pylab as plt
# Now import file from data set and create a dataframe
Univ1=pd.read_excel("c:/360DG/Datasets/University_Clustering.xlsx")
Univ1.describe()
#We have one column "State" which really not useful we will drop it
Univ=Univ1.drop(["State"],axis=1)
# we know that there is scale difference among the columns,which we have to remove
#either by using normalization or standardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Now apply this normalization function to Univ datframe for all the rows and column from 1 until end
    
df_norm=norm_func(Univ.iloc[:,1:])
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

# applying agglomerative clustering choosing 5 as clusters from dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity="euclidean").fit(df_norm)
# apply labels to the clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#Assign this series to Univ Dataframe as column and name the column as "clust"
Univ['clust']=cluster_labels
# we want to relocate the column 7 to 0 th position
Univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]
#now check the Univ1 dataframe
Univ1.iloc[:,2:].groupby(Univ1.clust).mean()
Univ1.to_csv("University.csv",encoding="utf-8")
import os
os.getcwd()
