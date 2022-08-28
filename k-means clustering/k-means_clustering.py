# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:46:52 2022

@author: Dell
"""


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
# generate random numbers in the range 0 to 1 and with uniform probability of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create a datafram with 0 rows and 2 columns
df_xy=pd.DataFrame(columns=["X","Y"])
# assign the values of X and Y to these columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x="X",y="Y",kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


Univ1=pd.read_excel("c:/360DG/Datasets/University_Clustering.xlsx")
Univ1.describe()
Univ=Univ1.drop(["State"],axis=1)
# we know that there is scale difference among the columns,which we have to remove
#either by using normalization or standardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Now apply this normalization function to Univ datframe for all the rows and column from 1 until end
    
df_norm=norm_func(Univ.iloc[:,1:])
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
TWSS

plt.plot(k,TWSS,'ro-');plt.xlabel("No_of_clusters");plt.ylabel("Total_within_SS")
model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()
Univ.to_csv("kmeans_University.csv",encoding="utf-8")
import os
os.getcwd()
