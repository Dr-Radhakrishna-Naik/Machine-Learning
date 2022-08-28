# -*- coding: utf-8 -*-
"""
1.	Business Problem
1.1.1	What is the business objective?
To provide real-time package tracking for each shipment,
 FedEx uses one of the world's largest computer and telecommunications networks. 
 The company's couriers operate SuperTracker hand-held computers,
 to record the transit of shipments through the FedEx integrated network.
 1.1.2 FedEx's use of technology focuses on the customer, 
  rather than merely on remaining competitive.
  With FedEx, businesses can determine the status of their 
  packages at all possible locations along the delivery route 
  in real time. Customers can track packages 
  in three ways: via the FedEx Web site on the Internet, 
  by using FedEx Ship Manager at fedex.com, or FedEx WorldTM Shipping Software.
1.2.	Are there any constraints?
     Each package must be identifiable and trackable, 
     so the database must be able to store the location of the package 
     and its history of locations. 
     Locations include trucks, planes, airports, and warehouses.
     All this information must be updated in real time.

@author: Radhakrishna Naik
"""
#Explanation of the Dataset:
#Year: The Year the data was collected
#Month: The Month in which the data was collected
#DayofMonth: The day of the month
#DayofWeek: The day of Week
#ActualShipmentTime: The Actual time when the package was sent for shipment. (ex: 1955 means 19 hours and 55 minutes i.e 7:55 PM)
#PlannedShipmentTime: The time when the package should have been sent for shipment. (ex: 1955 means 19 hours and 55 minutes i.e 7:55 PM)
#PlannedDeliveryTime: The time when the package should be delivered. (ex: 1955 means 19 hours and 55 minutes i.e 7:55 PM)
#CarrierName: The name of the Carrier which carried the package. CarrierNum: The number of the Carrier which carried the package.
#PlannedTimeofTravel: The estimated time to reach from Source to Destination. ( in minutes) ShipmentDelay: The time by which the package was shipped late. (in minutes. Negative value indicates that the package was shipped early. Ex: 4 indicates that the package was shipped 4 minutes late, whereas, -4 indicates that the package was shipped 4 minutes early)
#Source: The place from which the package was shipped.
#Destination: The place at which the package was delivered.
#Distance: Distance between Source and Destination in miles.
#Delivery_Status: Whether it got delivered at right time or not. (Dependent Variable)

########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:/360DG/Datasets/fedex.csv")
df.to_pickle("fedex.bz2")
##########################EDA
df.dtypes
df.Actual_Shipment_Time.describe()
#The average shipment time is 1.33 
#min is 1 and max is 2.4
# data is left skewed
df.Planned_Shipment_Time.describe()
#The average is 1.32 
#min is 0 and max is 2350
df.Planned_Delivery_Time .describe()
#The average is 1495 
#min is 0 and max 2400
df.Planned_TimeofTravel.describe()
#The average is 129
#min is -25 and max is 1435
###########################
plt.hist(df.Actual_Shipment_Time)
#Data is normally distributed
plt.hist(df.Planned_Shipment_Time)
#Data is normally distributed
plt.hist(df.Planned_Delivery_Time)
#Data is normally distributed
plt.hist(df.Planned_TimeofTravel)
#Data is normally distributed and most of the data is present in proximity of the cell
plt.hist(df.Shipment_Delay)
#Data is normally distributed
plt.hist(df.Delivery_Status)
#Data is normally distributed and most of the data is present in proximity of the cell
###########################

plt.boxplot(df.Planned_Shipment_Time)
#There are no outliers
plt.boxplot(df.Planned_Delivery_Time)
#There are no outliers
plt.boxplot(df.Carrier_Num)
#There are no outliers
plt.boxplot(df.Planned_TimeofTravel)
#There are outliers
plt.boxplot(df.Shipment_Delay)
#There are  outliers


numeric_features=df.select_dtypes(include=[np.number])
numeric_features.columns
correlation=numeric_features.corr()
print(correlation['Delivery_Status'].sort_values(ascending=False),'\n')
#shipment delay is highly correlated with delivery status
#####
f,ax=plt.subplots(figsize=(14,12))
plt.figure(figsize=(15,10))
plt.title("Correlation of numeric features with delivery status",y=1,size=16)
sns.heatmap(correlation,square=True,vmax=0.8)
##shipment delay is highly correlated with delivery status
#####################
df.isnull().sum()
#There are null values in Actual_Shipment_Time ,Planned_TimeofTravel,Shipment_Delay  Delivery_Status 
##############
df['Delivery_Status'].value_counts()
########################
feature_categorical=[feature for feature in df.columns if df[feature].dtype=='O']
for feature in feature_categorical:
    pd.crosstab(df[feature],df['Delivery_Status']).plot(kind='bar',figsize=(30,20))
    ###########################################################
    ##########
 df.dtypes
 from sklearn.impute import SimpleImputer  
df['Planned_TimeofTravel'].isnull().sum()
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
df['Planned_TimeofTravel']=pd.DataFrame(mean_imputer.fit_transform(df[["Planned_TimeofTravel"]]))
    df['Planned_TimeofTravel'].isnull().sum()
    ##Data preprocessing
    df.dtypes
import seaborn as sns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Planned_TimeofTravel"])
df_t=winsor.fit_transform(df[["Planned_TimeofTravel"]])

sns.boxplot(df_t.Planned_TimeofTravel)

mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
df['Delivery_Status']=pd.DataFrame(mean_imputer.fit_transform(df[["Delivery_Status"]]))
    df['Delivery_Status'].isnull().sum()
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
df['Shipment_Delay']=pd.DataFrame(mean_imputer.fit_transform(df[["Shipment_Delay"]]))
    df['Shipment_Delay'].isnull().sum()
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
df['Actual_Shipment_Time']=pd.DataFrame(mean_imputer.fit_transform(df[["Actual_Shipment_Time"]]))
    df['Actual_Shipment_Time'].isnull().sum()
df.isnull().sum()
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Shipment_Delay"])
df_t=winsor.fit_transform(df[["Shipment_Delay"]])
sns.boxplot(df_t.Shipment_Delay)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Distance"])
df_t=winsor.fit_transform(df[["Distance"]])
sns.boxplot(df_t.Distance)

####################################
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
#This is model of label_encoder which is applied to all the object type columns
df['Carrier_Name']=label_encoder.fit_transform(df['Carrier_Name'])
df['Source']=label_encoder.fit_transform(df['Source'])
df['Destination']=label_encoder.fit_transform(df['Destination'])
#####################################
#There are several columns having data of different scale
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(df.iloc[:,1:15])
################################################
df1=df.iloc[:100000,:]
df_norm1=df_norm.iloc[:100000,:]

TWSS=[]
k=list(range(2,26))
# The values generated by TWSS are 24 and two get x and y values 24 by 24 ,range has been changed 2:26
#again restart the kernel and execute once
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm1)
    TWSS.append(kmeans.inertia_)
TWSS

plt.plot(k,TWSS,'ro-');plt.xlabel("No_of_clusters");plt.ylabel("Total_within_SS")

# from the plot it is clear that the TWSS is reducing from k=2 to 3 and 3 to 4 
#than any other change in values of k,hence k=3 is selected
model=KMeans(n_clusters=4)
model.fit(df_norm1)
model.labels_
mb=pd.Series(model.labels_)
df1['clust']=mb
df1=df1.iloc[:,[15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
df1.iloc[:,:].groupby(df1.clust).mean()

df1.to_csv("kmeans_Fedex_new.csv",encoding="utf-8")
import os
os.getcwd()
###########################################
#Let us sort the values according to cluster number
final_df = df1.sort_values(by=['clust'], ascending=True)
#reinitialize the index

final_df=final_df.reset_index(drop=True)
#let us split the dataframe as per the clusters
#0 th cluster data points
final_df1=final_df.iloc[0:28673,:]
#1 st cluster data points
final_df2=final_df.iloc[28674:48260,:]
#2 nd cluster data points
final_df3=final_df.iloc[48261:75745,:]
#Third cluster data points
final_df4=final_df.iloc[75746:100000,:]
#####################################
#Now let us apply KNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
fed_norm1=norm_func(final_df1)
# Training the model
#Before that,let us assign input and output columns
X=np.array(fed_norm1.iloc[:,3:15])
y=np.array(fed_norm1['Delivery_Status'])
#let us split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#Let us apply to knn model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred
#Evaluate the accuracy and applicability of the model
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)
pd.crosstab(pred,y_test,rownames = ['Actual'], colnames= ['Predictions'])
#Let us check the actual applicability of the model
#from confusion matrix 8 items are delivered but the model predicts it is not  delivered
#similarly actual 52 items are not delivered but which the model suggest it is delivered
#This is going to create big problem
##########
#Error on train data
pred_train=knn.predict(X_train)
accuracy_score(pred_train,y_train)
#0.6608187134502924
pd.crosstab(pred_train,y_train,rownames=['Actual'],colnames=['predicted'])
#Let us check the actual applicability of the model
#from confusion matrix 12 items are delivered but the model predicts it is not  delivered
#similarly actual 122 items are not delivered but which the model suggest it is delivered
#This is going to create big problem
##############################################
#Tunning of the model
#For selection of optimum value of k
acc=[]
#Let us run KNN on values 3,50 in step of 2 so that next value will be odd
for i in range(3,50,2):
    knn1=KNeighborsClassifier(n_neighbors=i)
    knn1.fit(X_train,y_train)
    train_acc=np.mean(knn1.predict(X_train)==y_train)
    test_acc=np.mean(knn1.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])
#To plot the graph of accuracy of training and testing
    import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")    
plt.plot(np.arange(3,50,2),[i[1]for i in acc],"bo-")
#from the plot k=9 is value where test accuracy and train accuracy are equal
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred
#Evaluate the accuracy and applicability of the model
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)
pd.crosstab(pred,y_test,rownames = ['Actual'], colnames= ['Predictions'])
#Let us check the actual applicability of the model
#from confusion matrix 3 items are delivered but the model predicts it is not  delivered
#similarly actual 63 items are not delivered but which the model suggest it is delivered
#This is going to create big problem
#################################
import pickle
pickle.dump(knn,open("knn_model.pickle.dat","wb"))
loaded_model=pickle.load(open("knn_model.pickle.dat","rb"))
###########################################################
#Now let us apply Naive Bayes to final_df2

#Now let us apply normalization function
def norm_funct(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
fed_norm2=norm_funct(final_df2)
# Training the model
#Before that,let us assign input and output columns
X=np.array(fed_norm2.iloc[:,3:15])
y=np.array(final_df2['Delivery_Status'])
#let us split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
##	Model Building
#Build the model on the scaled data (try multiple options).
#Build a Naïve Bayes model.
#Like MultinomialNB, this classifier is suitable for discrete data. The difference is that while MultinomialNB works with occurrence counts, 
#BernoulliNB is designed for binary/boolean features.
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()

classifier_mb.fit(X_train,y_train)
#Let us now evaluate on test data
test_pred_m=classifier_mb.predict(X_test)
##Accuracy of the prediction
accuracy_test_m=np.mean(test_pred_m==y_test)
accuracy_test_m
###Let us now check confusion matrix
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m,y_test)
pd.crosstab(test_pred_m,y_test)
######################################################
import pickle
pickle.dump(knn,open("classifier_mb_model.pickle.dat","wb"))
loaded_model=pickle.load(open("classifier_mb_model.pickle.dat","rb"))
##########################################################################
###let us apply decision tree to final_df3
##There are several columns having different scale
#Normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm3=norm_func(final_df3.iloc[:,3:16])
####################################################
#Let us check that how many unique values are the in the output column
df_norm3["Delivery_Status"].unique()
df_norm3["Delivery_Status"].value_counts()

###########################
#let us assign input features as predictors and output as target
colnames=list(df_norm3.columns)
predictors=colnames[:12]
target=colnames[12]
######################################
#Splitting data into train and Test
from sklearn.model_selection import train_test_split
train,test=train_test_split(df_norm3,test_size=0.2)
##############################################
#model bulding
from sklearn.tree import DecisionTreeClassifier as DT
model=DT(criterion='entropy')
model.fit(train[predictors],train[target])
preds=model.predict(test[predictors])
pd.crosstab(test[target],preds)
np.mean(preds==test[target])
#Let us check the accuracy on training dataset
preds=model.predict(train[predictors])
np.mean(preds==train[target])
#########################################################
import pickle
pickle.dump(knn,open("DT_model.pickle.dat","wb"))
loaded_model=pickle.load(open("DT_model.pickle.dat","rb"))
