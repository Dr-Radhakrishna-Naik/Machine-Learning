# -*- coding: utf-8 -*-
"""
1.	A cloth manufacturing company is interested to know 
about the different attributes contributing to high sales. Build a decision tree & 
random forest model with Sales as target variable 
1.	Business Problem
1.1.	What is the business objective?
Successful fashion marketing depends on understanding 
consumer desire and responding with appropriate products.
 Marketers use sales tracking data, 
 attention to media coverage, focus groups, 
 and other means of ascertaining consumer preferences 
 to provide feedback to designers and manufacturers about
 the type and quantity of goods to be produced. Marketers are thus responsible for identifying and defining a fashion producer’s target customers and for responding to the preferences of those customers.
1.1.	Are there any constraints?
    Technology is the key to facing the challenge of greater competition from imports. 
@author: Radhakrishna Naik
"""
#Sales: unit sales in thousands

#CompPrice: price charged by competitor at each location

#Income: community income level in 1000s of dollars

#Advertising: local ad budget at each location in 1000s of dollars

#Population: regional pop in thousands

#Price: price for car seats at each site

#ShelveLoc: Bad, Good or Medium indicates quality of shelving location

#Age: age level of the population

#Education: ed level at location

#Urban: Yes/No

#US: Yes/No
####################################
 import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
cloth=pd.read_csv("c:/360DG/Datasets/Company_Data.csv")
############################
#exploratory data analysis
cloth.dtypes
cloth.describe()
#Average Sale is 7.4
#minimum is 0 and max is 16.27
#sales data is right skewed
#minimum compitators price 77 max is 175
#Average sale 124.97
#average community income is 68.65
#minimum is 21 and max 120
plt.hist(cloth.Sales)
#Data is normally distributed
plt.hist(cloth.CompPrice)
#Data is normally distributed
plt.hist(cloth.Income)
#Data has got kurtosis
plt.hist(cloth.Age)
###########
#let us check outliers
plt.boxplot(cloth.Sales)
#There are outliers in Sales data
plt.boxplot(cloth.CompPrice)
#There are outliers
plt.boxplot(cloth.Income)
#There are no outliers
plt.boxplot(cloth.Age)
#There are no outliers
cloth.dtypes
plt.boxplot(cloth.Education)
#There are no outliers
cloth.isnull().sum
#################################
#Data preprocessing
#ShelveLoc,Urban,Us data is object type
bins=[0,5,10,15,20]
group_name=['low','ave','good','better']
cloth['Sales_cat']=pd.cut(cloth['Sales'],bins,labels=group_name)
cloth.isnull().sum()
cloth.dropna()
cloth.isnull().sum()
from sklearn.impute import SimpleImputer
mode_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
cloth['Sales_cat']=pd.DataFrame(mode_imputer.fit_transform(cloth[["Sales_cat"]]))
cloth['Sales_cat'].isna().sum()

from sklearn.preprocessing import LabelEncoder
import seaborn as sns
lb=LabelEncoder()
cloth["ShelveLoc"]=lb.fit_transform(cloth["ShelveLoc"])
cloth["Urban"]=lb.fit_transform(cloth["Urban"])
cloth["US"]=lb.fit_transform(cloth["US"])
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Sales"])
cloth_t=winsor.fit_transform(cloth[["Sales"]])
sns.boxplot(cloth_t.Sales)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["CompPrice"])
cloth_t=winsor.fit_transform(cloth[["CompPrice"]])
sns.boxplot(cloth_t.CompPrice)
##There are several columns having different scale
#Normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
cloth_norm=norm_func(cloth.iloc[:,1:11])
####################################################
#Let us check that how many unique values are the in the output column
cloth["Sales_cat"].unique()
cloth["Sales_cat"].value_counts()


###########################
#let us assign input features as predictors and output as target
colnames=list(cloth.columns)
predictors=colnames[1:11]
target=colnames[11]
##################################
#Splitting data into train and Test
from sklearn.model_selection import train_test_split
train,test=train_test_split(cloth,test_size=0.2)
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
#6.	Write about the benefits/impact of the solution - in 
#what way does the business (client) benefit from the solution provided?
#The use of decision tree-based data mining to establish Sales is advantageous
 #because it can (1) allow for a wider coverage of features matrix with a fewer number of steps, 
 #(2) generate regular sales patterns and rules, and (3) provide manufacturers with reference points to facilitate production. 
 #The newly developed sales system can provide garment manufacturers with insights, design development, pattern grading, and market analysis.
 #Moreover, when production plans can be made more realistic, inventory costs due to mismatches can be minimized.