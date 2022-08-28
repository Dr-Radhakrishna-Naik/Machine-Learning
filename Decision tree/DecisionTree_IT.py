# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:04:47 2022

@author: Radhakrishna Naik
"""

# -*- coding: utf-8 -*-
"""
3.	Build a Decision Tree & Random Forest model on the fraud data
. Treat those who have taxable_income <= 30000 as Risky 
and others as Good 
1.	Business Problem
  1.1.	What is the business objective?
 1.1.1 Build  Analytics  Model  based  on  industry-oriented 
 indicators and utilize knowledge base of organization;
 1.1.2. Employ effectively  technology including  platforms,
 algorithms and proven concepts; 
 1.1.3. Pre-process Data turning it to Quality Data - 
 Quality results require high quality data; 
 1.1.4. Utilize  traditional  fraud  detection  skills  
 in  Data Analytics changing scale, dimension and depth.
 
1.2.	Are there any constraints?
    1.2.1 The  problem  of  Data  Analytics  being  not  employed
 adequately is in the lack of understanding by Income tax authorities 
 what Advanced Data Analytics can do for combatting fraudulent actions and how effective 
 it can be on this particular task.

@author: Radhakrishna Naik
"""
# Data Description
#Undergrad          object
#Marital.Status     object
#Taxable.Income      int64
#City.Population     int64
#Work.Experience     int64
#Urban              object


####################################
 import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
IT=pd.read_csv("c:/360DG/Datasets/Fraud_check.csv")
############################
#exploratory data analysis
IT.dtypes
IT.describe()
#Average  taxable income 55208.375000
#minimum is 10003.000000 and max is 99619.000000
#Average  City population 108747
#minimum is 25779 and max is 199778
#Average  Work Experience 15.55
#minimum is 0 and max is 30
#######################################
#Let us rename the columns
new_names = ['UG','Mar_stat','Tax_income', 'City_pop', 'Exp', 'Urban']
df = pd.read_csv(
    "c:/360DG/Datasets/Fraud_check.csv", 
    names=new_names,           # Rename columns
    header=0,                  # Drop the existing header row
    usecols=[0,1,2,3,4,5],       # Read the first 5 columns
    )
df.dtypes
plt.hist(df.Tax_income)
#Data is almost uniformly distributed
plt.hist(df.Exp)
#Data is almost uniformly distributed

plt.hist(df.City_pop)
#Data is almost uniformly distributed

###########
#let us check outliers
plt.boxplot(df.Tax_income)
#There are  no outliers 
plt.boxplot(df.Exp)
#There are no outliers
plt.boxplot(df.City_pop)
#There are no outliers

df.isnull().sum()
#################################
#Data preprocessing
#ShelveLoc,Urban,Us data is object type
bins = [10002,30000,99620]
group_name=["Risky", "Good"]
df['Tax_income']=pd.cut(df['Tax_income'],bins,labels=group_name)
from sklearn.impute import SimpleImputer
mode_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
cloth['Sales_cat']=pd.DataFrame(mode_imputer.fit_transform(cloth[["Sales_cat"]]))
cloth['Sales_cat'].isna().sum()

from sklearn.preprocessing import LabelEncoder
import seaborn as sns
lb=LabelEncoder()
df["UG"]=lb.fit_transform(df["UG"])
df["Mar_stat"]=lb.fit_transform(df["Mar_stat"])
df["Urban"]=lb.fit_transform(df["Urban"])
##There are several columns having different scale
#Normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df=df[['Tax_income','UG','Mar_stat','City_pop','Exp','Urban']]
df_norm=norm_func(df.iloc[:,1:6])
####################################################
#Let us check that how many unique values are the in the output column
df["Tax_income"].unique()
df["Tax_income"].value_counts()


###########################
#let us assign input features as predictors and output as target
colnames=list(df.columns)
predictors=colnames[1:5]
target=colnames[0]
##################################
#Splitting data into train and Test
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2)
##############################################
#model bulding
from sklearn.tree import DecisionTreeClassifier as DT
model=DT(criterion='entropy')
model.fit(train[predictors],train[target])
preds=model.predict(test[predictors])
pd.crosstab(test[target],preds)
np.mean(preds==test[target])
####
#Let us check the accuracy on training dataset
preds=model.predict(train[predictors])
np.mean(preds==train[target])
################################################################
# Now let us try for Random forest tree
from sklearn.ensemble import RandomForestClassifier
rand_for=RandomForestClassifier(n_estimators=500,n_jobs=1,random_state=42)
rand_for.fit(train[predictors],train[target])
from sklearn.metrics import accuracy_score,confusion_matrix
preds=model.predict(test[predictors])
pd.crosstab(test[target],preds)
np.mean(preds==test[target])
################################################################
#Let us check the accuracy on training dataset
preds=model.predict(train[predictors])
np.mean(preds==train[target])
pd.crosstab(train[target],preds)
##################################################################
##this is again overfit model
rand_for=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20,criterion='gini')
rand_for.fit(train[predictors],train[target])
from sklearn.metrics import accuracy_score,confusion_matrix
preds=model.predict(test[predictors])
pd.crosstab(test[target],preds)
np.mean(preds==test[target])


#6.	Write about the benefits/impact of the solution - in 
#what way does the business (client) benefit from the solution provided?
# Continuously  improve  Fraud Detection Models and Indicators 
 #According to  survey only 3% of  fraud cases 
 #were detected using fraud-focused analytics, 
 #while  44%  were  found  based  on  intuition  or whistle-blower mechanism
#The use of decision tree-based data mining to establish prediction of frauds
 