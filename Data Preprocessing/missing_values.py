# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:57:15 2022

@author: Dell
"""


import numpy as np
import pandas as pd
df=pd.read_csv("c:/360DG/Datasets/Claimants.csv")
df.isna().sum()
#CLMSEX=12,CLMINSUR=41,SEATBELT=48,CLMAGE=189 missing values
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
df['CLMSEX']=pd.DataFrame(mean_imputer.fit_transform(df[['CLMSEX']]))
df['CLMSEX'].isna().sum()
df['CLMINSUR']=pd.DataFrame(mean_imputer.fit_transform(df[['CLMINSUR']]))
df['CLMINSUR'].isna().sum()
df['SEATBELT']=pd.DataFrame(mean_imputer.fit_transform(df[['SEATBELT']]))
df['SEATBELT'].isna().sum()
df['CLMAGE']=pd.DataFrame(mean_imputer.fit_transform(df[['CLMAGE']]))
df['CLMAGE'].isna().sum()
# Now check whether there are outliers in each of these columns
import seaborn as sns
sns.boxplot(df.CLMAGE)
# yes there are outliers 
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
df['CLMSEX']=pd.DataFrame(median_imputer.fit_transform(df[['CLMSEX']]))
df['CLMSEX'].isna().sum()
sns.boxplot(df.CLMINSUR)
# yes there are outliers in CLMINSUR
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
df['CLMINSUR']=pd.DataFrame(median_imputer.fit_transform(df[['CLMINSUR']]))
df['CLMINSUR'].isna().sum()
sns.boxplot(df.SEATBELT)
# yes there are outliers
df['SEATBELT']=pd.DataFrame(median_imputer.fit_transform(df[['SEATBELT']]))
df['SEATBELT'].isna().sum()
sns.boxplot(df.CLMAGE)
# yes there are outliers
df['CLMAGE']=pd.DataFrame(median_imputer.fit_transform(df[['CLMAGE']]))
df['CLMAGE'].isna().sum()
