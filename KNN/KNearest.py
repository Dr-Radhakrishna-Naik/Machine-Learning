# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:42:44 2022

@author: Dell
"""
import pandas as pd
import numpy as np
wbcd=pd.read_csv("C:/360DG/Datasets/wbcd.csv")
##let us prepare the data
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='B','Benign',wbcd['diagnosis'])
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='M','Malignant',wbcd['diagnosis'])

##In our dataframe 0 th column is id column ,which not useful
wbcd=wbcd.iloc[ :,1:32]

# Normalization function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Now let us carry out normalization of data frame excluding diagnosis column
    
wbcd_n=norm_func(wbcd.iloc[ :,1:])

wbcd_n.describe()
#Now let us assign the input as X and label or output as Y
X=np.array(wbcd_n.iloc[:,:]) ##All normalized values excluding diagnosis column
Y=np.array(wbcd['diagnosis'])

##Now let us split the the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,Y_train)
pred=knn.predict(X_test)
pred

##Now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,pred))
pd.crosstab(Y_test,pred,rownames=['Actual'],colnames=['Predictions'])

# error on train data
pred_train=knn.predict(X_train)
print(accuracy_score(Y_train,pred_train))

###for selection of optimum value of k ,we need to test k-nn accross valuse of k
acc=[]
##running KNN algorith for 3 to 50 nearest neighbors(odd numbers)
for i in range (3,50,2):
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,Y_train)
    train_acc=np.mean(neigh.predict(X_train)==Y_train)
    test_acc=np.mean(neigh.predict(X_test)==Y_test)
    acc.append([train_acc,test_acc])
    
###To plot these values on line graph
    import matplotlib.pyplot as plt
    plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
    plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
