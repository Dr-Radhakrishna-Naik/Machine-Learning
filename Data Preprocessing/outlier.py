# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 12:25:22 2022

@author: Dell
"""


import pandas as pd

#pip install feature_engine --user
from feature_engine.outliers import Winsorizer

import numpy as np
import seaborn as sns
df=pd.read_csv("c:/360DG/Datasets/Boston.csv")
df.dtypes
sns.boxplot(df.tax)
IQR=df['tax'].quantile(0.75)-df['tax'].quantile(0.25)
lower_limit=df['tax'].quantile(0.25)-(IQR*1.5)
upper_limit=df['tax'].quantile(0.75)+(IQR*1.5)
# trimming technique
outlier_df=np.where(df['tax']>upper_limit,True,np.where(df['tax']<lower_limit,True,False))
df_trimmed=df.loc[~(outlier_df),]
df.shape
df_trimmed.shape
sns.boxplot(df_trimmed.tax)
###
sns.boxplot(df.crim)
IQR=df['crim'].quantile(0.75)-df['crim'].quantile(0.25)
lower_limit=df['crim'].quantile(0.25)-(IQR*1.5)
upper_limit=df['crim'].quantile(0.75)+(IQR*1.5)
# trimming technique
outlier_df=np.where(df['crim']>upper_limit,True,np.where(df['crim']<lower_limit,True,False))
df_trimmed=df.loc[~(outlier_df),]
df.shape
df_trimmed.shape
sns.boxplot(df_trimmed.crim)

# Replace
df['df_crim']=pd.DataFrame(np.where(df['crim']>upper_limit,upper_limit,np.where(df['crim']<lower_limit,lower_limit,df['crim'])))
sns.boxplot(df.df_crim)
df.dtypes
sns.boxplot(df.crim)
### for zn
outlier_df=np.where(df['zn']>upper_limit,True,np.where(df['zn']<lower_limit,True,False))
df_trimmed=df.loc[~(outlier_df),]
df.shape
df_trimmed.shape
sns.boxplot(df_trimmed.zn)
###
sns.boxplot(df.zn)
IQR=df['zn'].quantile(0.75)-df['zn'].quantile(0.25)
lower_limit=df['zn'].quantile(0.25)-(IQR*1.5)
upper_limit=df['zn'].quantile(0.75)+(IQR*1.5)
# trimming technique
outlier_df=np.where(df['zn']>upper_limit,True,np.where(df['zn']<lower_limit,True,False))
df_trimmed=df.loc[~(outlier_df),]
df.shape
df_trimmed.shape
sns.boxplot(df_trimmed.zn)

# Replace
df['df_zn']=pd.DataFrame(np.where(df['zn']>upper_limit,upper_limit,np.where(df['zn']<lower_limit,lower_limit,df['zn'])))
sns.boxplot(df.df_zn)
df.dtypes
sns.boxplot(df.zn)
sns.boxplot(df.zn)
IQR=df['zn'].quantile(0.75)-df['zn'].quantile(0.25)
lower_limit=df['zn'].quantile(0.25)-(IQR*1.5)
upper_limit=df['zn'].quantile(0.75)+(IQR*1.5)
# trimming technique
sns.boxplot(df.age)
## for age there are no outliers
df.dtypes
sns.boxplot(df.indus)
# for indus there are no outliers
# for chas there are all values null
sns.boxplot(df.nox)
# for nox there are no putliers
sns.boxplot(df.rm)
# for rm there are outliers
IQR=df['rm'].quantile(0.75)-df['rm'].quantile(0.25)
lower_limit=df['rm'].quantile(0.25)-(IQR*1.5)
upper_limit=df['rm'].quantile(0.75)+(IQR*1.5)
# trimming technique
outlier_df=np.where(df['rm']>upper_limit,True,np.where(df['rm']<lower_limit,True,False))
df_trimmed=df.loc[~(outlier_df),]
df.shape
df_trimmed.shape
sns.boxplot(df_trimmed.rm)

# Replace
df['df_rm']=pd.DataFrame(np.where(df['rm']>upper_limit,upper_limit,np.where(df['rm']<lower_limit,lower_limit,df['rm'])))
sns.boxplot(df.df_rm)
df.dtypes
sns.boxplot(df.rm)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['rm'])
df_t=winsor.fit_transform(df[['rm']])
sns.boxplot(df_t.rm)
######################
###age
sns.boxplot(df.age)
###There are no outliers in age
#################
####dis
sns.boxplot(df.dis)
#### there are outliers in dis
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['dis'])
df_t=winsor.fit_transform(df[['dis']])
sns.boxplot(df_t.dis)
sns.boxplot(df.rad)
#### There are no outliers in rad
sns.boxplot(df.ptratio)
########There are outliers in ptratio
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['ptratio'])
df_t=winsor.fit_transform(df[['ptratio']])
sns.boxplot(df_t.ptratio)
sns.boxplot(df.ptratio)
sns.boxplot(df.black)
#### there are outliers in black
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['black'])
df_t=winsor.fit_transform(df[['black']])
sns.boxplot(df_t.black)
sns.boxplot(df.black)
sns.boxplot(df.black)
#############
#####lstat
sns.boxplot(df.lstat)
###### there are outliers
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['lstat'])
df_t=winsor.fit_transform(df[['lstat']])
sns.boxplot(df_t.lstat)
sns.boxplot(df.lstat)
##################################
#####medv
sns.boxplot(df.medv)
#####there are outliers
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['medv'])
df_t=winsor.fit_transform(df[['medv']])
sns.boxplot(df_t.medv)
sns.boxplot(df.medv)
#######################################
###########df_crim
sns.boxplot(df.df_crim)
####There are no outliers
######df_zn
sns.boxplot(df.df_zn)
#############There are no outliers
