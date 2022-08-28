# -*- coding: utf-8 -*-
"""
Divide the diabetes data into train and test datasets 
and build a Random Forest and Decision Tree model 
with Outcome as the output variable. 
1.	Business Problem
1.1.	What is the business objective?
  1.1.1 With the development of living standards, diabetes is increasingly
 common in people’s daily life. Therefore, how to quickly and accurately
 diagnose and analyze diabetes is a topic worthy studying. 
  1..1.2 In medicine, the diagnosis of diabetes is according to fasting blood glucose,
 glucose tolerance, and random blood glucose levels. 
 The earlier diagnosis is obtained, the much easier we can control it. Machine learning can help people make a preliminary judgment about diabetes mellitus according to their daily physical examination data, and it can serve as a reference for doctors
  
1.1.	Are there any constraints?
    For machine learning method, how to select the valid features 
    and the correct classifier are the most important constraints.
Several constraints were placed on the selection of these 
instances from a larger database. In particular, all patients
 here are females at least 21 years old of Pima Indian heritage.
@author: Radhakrishna Naik
"""
#Data description
#Pregnancies:Number of times pregnant int64	
#Glucose	:Plasma glucose concentration int64
#BloodPressure	:Diastolic blood pressure int64
#SkinThickness	:Triceps skin fold thickness  int64
#Insulin	:2-Hour serum insulin int64
#BMI	    :Body mass index  float
#DiabetesPedigreeFunction	
#Age	
#Outcome
####################################
 import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
diabetics=pd.read_csv("c:/360DG/Datasets/Diabetes.csv")
############################
#exploratory data analysis
diabetics.dtypes
diabetics.describe()
#Average number of times pregnant is 3.84
#minimum is 0 and max is 17
#number of times pregnant data is right skewed
#minimum age 21 max is 81
#Average age 33.24
#Age is right skewed
########################################
diabetics.dtypes

new_names = ['Pregnancies', 'Glucose', 'BP', 'Skin_thickness', 'Insulin','BMI','D_pedigree','Age','Outcome']
df = pd.read_csv(
    "c:/360DG/Datasets/Diabetes.csv", 
    names=new_names,           # Rename columns
    header=0,                  # Drop the existing header row
    usecols=[0,1,2,3,4,5,6,7,8],       # Read the first 5 columns
)
df.dtypes
plt.hist(df.Pregnancies)
#Data is normally distributed,right skewed
plt.hist(df.Glucose)
#Data is normally distributed,slight left skewed
plt.hist(df.BP)
#Data is normally distributed
plt.hist(df.Skin_thickness)
#Data is normal distributed
###########
df.dtypes
#let us check outliers
plt.boxplot(df.Pregnancies)
#There are outliers in Pregnancies data
plt.boxplot(df.Glucose)
#There are outliers
plt.boxplot(df.BP)
#There are  outliers
plt.boxplot(df.Skin_thickness)
#There are outliers
plt.boxplot(df.Insulin)
#There are several outliers
plt.boxplot(df.BMI)
#There are several outliers
plt.boxplot(df.D_pedigree)
#There are several outliers
plt.boxplot(df.Age)
#There are several outliers

df.isnull().sum()
#################################
df.dtypes
import seaborn as sns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Pregnancies"])
df_t=winsor.fit_transform(df[["Pregnancies"]])
sns.boxplot(df_t.Pregnancies)

df.dtypes
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Glucose"])
df_t=winsor.fit_transform(df[["Glucose"]])
sns.boxplot(df_t.Glucose)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["BP"])
df_t=winsor.fit_transform(df[["BP"]])
sns.boxplot(df_t.BP)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Skin_thickness"])
df_t=winsor.fit_transform(df[["Skin_thickness"]])
sns.boxplot(df_t.Skin_thickness)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Insulin"])
df_t=winsor.fit_transform(df[["Insulin"]])
sns.boxplot(df_t.Insulin)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["BMI"])
df_t=winsor.fit_transform(df[["BMI"]])
sns.boxplot(df_t.BMI)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["Age"])
df_t=winsor.fit_transform(df[["Age"]])
sns.boxplot(df_t.Age)
###################################################
##There are several columns having different scale
#Normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(df.iloc[:,:8])
####################################################
#Let us check that how many unique values are the in the output column
df["Outcome"].unique()
df["Outcome"].value_counts()


###########################
#let us assign input features as predictors and output as target
colnames=list(df.columns)
predictors=colnames[:8]
target=colnames[8]
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
#6.	Write about the benefits/impact of the solution - in 

#what way does the business (client) benefit from the solution provided?
#This  will portray how data related to diabetes can be leveraged
# to predict if a person has diabetes or not. More specifically, this will focus on how machine learning 
# can be utilized to predict diseases such as diabetes.
#The use of decision tree-based data mining to establish prediction of adiabetics is advantageous
 #because it can (1) allow for a wider coverage of features matrix with a fewer number of steps, 
 #(2) generate regular predictions patterns and rules, and (3) provide doctors with reference points to facilitate the treatment. 
 #The newly developed sales system can provide garment manufacturers with insights, design development, pattern grading, and market analysis.
 #Moreover, when production plans can be made more realistic, inventory costs due to mismatches can be minimized.