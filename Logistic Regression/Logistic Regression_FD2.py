# -*- coding: utf-8 -*-
"""
	4.	It is vital for banks that customers put in long term fixed deposits
    as they use it to pay interest to customers and it is not viable 
    to ask every customer if they will put in a long-term deposit or not. 
    So, build a Logistic Regression model to predict whether a customer 
    will put in a long-term fixed deposit or not based on the different variables
    given in the data. The output variable in the dataset is Y which 
    is binary. 
 1.	Business Problem
1.1.	What is the business objective?
      1.1.1TMarketing is a process by which companies create value for 
      customers and build strong customer relationships in order to capture 
      value from customers in return. Marketing campaigns are characterized
      by focusing on the customer needs and their overall satisfaction. 
      Nevertheless, there are different variables that determine whether 
      a marketing campaign will be successful or not. There are certain 
      variables that we need to take into consideration when making a
      marketing campaign.
      1.1.2 Predicting-whether-the-customer-will-subscribe-to-Term-Deposits
1.2.	Are there any constraints?
        Although huge set of varibles have been given
        Varibles may be having poor collinearity.we need to check 
        collinearity of each with target
@author: Radhakrishna Naik
"""
#Dataset
#0   ID         13564 non-null  int64 
# 1   age        (numeric)  int64 
# 2   job        type of job (categorical: ‘admin.’,’blue-collar’,’entrepreneur’,’housemaid’,’management’,’retired’,’self-employed’,’services’,’student’,’technician’,’unemployed’,’unknown’)  
# 3   marital    marital status (categorical: ‘divorced’,’married’,’single’,’unknown’; note: ‘divorced’ means divorced or widowed)
# 4   education  (categorical: ‘basic.4y’,’basic.6y’,’basic.9y’,’high.school’,’illiterate’,’professional.course’,’university.degree’,’unknown’)
# 5   default    has credit in default? (categorical: ‘no’,’yes’,’unknown’)
# 6   balance    13564 non-null  int64 
# 7   housing    has housing loan? (categorical: ‘no’,’yes’,’unknown’)
# 8   loan       has personal loan? (categorical: ‘no’,’yes’,’unknown’)
# 9   contact    contact communication type (categorical: ‘cellular’,’telephone’)
# 10  day        13564 non-null  int64 
# 11  month      last contact month of year (categorical: ‘jan’, ‘feb’, ‘mar’, …, ‘nov’, ‘dec’)
# 12  duration   last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y=’no’). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# 13  campaign   number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 14  pdays      number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
#15  previous   number of contacts performed before this campaign and for this client (numeric)
 #16  poutcome   outcome of the previous marketing campaign (categorical: ‘failure’,’nonexistent’,’success’)
 #Dataset given is already processed 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
fd=pd.read_csv("c:/360DG/Datasets/bank_data.csv")
fd.dtypes
fd.columns="age","default","balance","housing","loan","duration","campaign","pdays","previous","poutfailure","poutother","poutsuccess","poutunknown","con_cellular","con_telephone","con_unknown","divorced","married","single","jobadmin","joblue_collar","jobentrepreneur","johousemaid","job_mgt","job_retired","joself_employed","jobservices","jostudent","jotechnician","job_unemployed","job_unknown","target"
fd.isnull().sum()
#There are zero null values
fd.describe()
#Avrage age is 40.93 ,min is 18 and max is 95
#The type of job the customers have. Let’s call the count plot function defined earlier to plot the count plot of the job feature.
import seaborn as sns
plt.figure(1, figsize=(16, 10))
sns.countplot(fd['jobadmin'])
sns.countplot(fd['joblue_collar'])
sns.countplot(fd['jobentrepreneur'])
sns.countplot(fd['johousemaid'])
sns.countplot(fd['job_mgt'])
sns.countplot(fd['job_retired'])
sns.countplot(fd['joself_employed'])
sns.countplot(fd['jobservices'])
sns.countplot(fd['jostudent'])
sns.countplot(fd['jobservices'])
sns.countplot(fd['jotechnician'])
sns.countplot(fd['job_unemployed'])
sns.countplot(fd['job_unknown'])
#There are more customers working as admin than any other profession.
#Now let us check the marital status
plt.figure(1, figsize=(16, 10))
sns.countplot(fd['divorced'])
sns.countplot(fd['married'])
sns.countplot(fd['single'])
#majority customers are divorced
#Default: Denotes if the customer has credit in default or not. The categories are yes, no and unknown
plt.figure(1, figsize=(16, 10))
sns.countplot(fd['default'])
#housing: Denotes if the customer has a housing loan. Three categories are ‘no’, ’yes’, ’unknown’.
sns.countplot(fd['housing'])
#poutcome: This feature denotes the outcome of the previous marketing campaign.
sns.countplot(fd['poutfailure'])
sns.countplot(fd['poutother'])
sns.countplot(fd['poutsuccess'])
#There are column names having spaces ,let us rename the columns
tc = fd.corr()
tc

fig,ax= plt.subplots()
fig.set_size_inches(40,20)
sns.heatmap(tc, annot=True, cmap='YlGnBu')
#From heat map it is clear that only duration,poutsuccess,con_cellular,pdays are highly correlated with target,remaining can be dropped
#We can infer that duration of the call,(outcome of the previous marketing campaign)poutsuccess is highly correlated with 
#the target variable. As the duration of the call is more, there are higher chances that the client is showing interest in the term deposit and hence there are higher chances that the client will subscribe to term deposit.
#con_cellular and pdays are upto some extent correlated
fd=fd.drop(["age","default","balance","housing","loan","campaign","previous","poutfailure","poutother","poutunknown","con_telephone","con_unknown","divorced","married","single","jobadmin","joblue_collar","jobentrepreneur","johousemaid","job_mgt","job_retired","joself_employed","jobservices","jostudent","jotechnician","job_unemployed","job_unknown"],axis=1)
#
fd=fd.drop(["con_cellular","pdays"],axis=1)

#Many columns have different scale values let us apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
fd1=norm_func(fd.iloc[:,:])
fd1.isna().sum()
#There are no null values
fd1.dtypes
######################
#model bulding
logit_model=sm.logit('target ~ duration+poutsuccess',data=fd1).fit()
logit_model.summary()
logit_model.summary2()
#let us go for prediction
pred=logit_model.predict(fd1.iloc[:,:2])
#####################
#To derive ROC curve
#ROC curve has tpr on y axis and fpr on x axis,ideally,tpr must be high
#fpr must be low
fpr,tpr,thresholds=roc_curve(fd1.target,pred)
#To identify optimum threshold
optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=thresholds[optimal_idx]
optimal_threshold
#0.1059 ,by default you can take 0.5 value as a threshold
#Now we want to identify if new value is given to the model,it will
#fall in which region 0 or 1,for that we need to derive ROC curve
#To draw ROC curve
import pylab as pl
i=np.arange(len(tpr))
roc=pd.DataFrame({'fpr':pd.Series(fpr,index=i),'tpr':pd.Series(tpr,index=i),'1-fpr':pd.Series(1-fpr,index=i),'tf':pd.Series(tpr-(1-fpr),index=i),'thresholds':pd.Series(thresholds,index=i)})
#plot ROC curve
plt.plot(fpr,tpr)
plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc=auc(fpr,tpr)
print("Area under the curve %f"%roc_auc)
#############
#Now let us add prediction column in dataframe
fd1["pred"]=np.zeros(45211)
fd1.loc[pred>optimal_threshold,"pred"]=1
#if predicted value is greater than optimal threshold then change pred column as 1
#Classification report
classification=classification_report(fd1["pred"],fd1["target"])
classification
###################################
#splitting the data into train and test data
train_data,test_data=train_test_split(fd1,test_size=0.3)
#model building using 
model=sm.logit('target ~ duration+poutsuccess',data=train_data).fit()
model.summary()
model.summary2()
#AIC is 17382.8812
#prediction on test data
test_pred=model.predict(test_data)
test_data["test_pred"]=np.zeros(13564)
#taking threshold value as optimal threshold value
test_data.loc[test_pred>optimal_threshold,"test_pred"]=1
#Confusion_matrix
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.target)
confusion_matrix
accuracy_test=(10033+1095)/13564
accuracy_test
#0.8204
#Classification report
classification_test=classification_report(test_data["test_pred"],test_data["target"])
classification_test
#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(test_data["target"],test_pred)
#plot of ROC
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test
###prediction on train data
train_pred=model.predict(train_data.iloc[:,:2])
#creating new column
train_data["train_pred"]=np.zeros(31647)
train_data.loc[train_pred>optimal_threshold,"train_pred"]=1
#confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred,train_data.target)
confusion_matrix
####
#Accuracy test
accuracy_train=(23378+2561)/31647
accuracy_train
#classification report
classification_train=classification_report(train_data.train_pred,train_data.target)
classification_train
##########
#ROC_AUC curve
roc_auc_train=metrics.auc(fpr,tpr)
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
#6.	The benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
#The objective of this case study was to predict whether a customer will subscribe to a term deposit or not given the data of the customer.