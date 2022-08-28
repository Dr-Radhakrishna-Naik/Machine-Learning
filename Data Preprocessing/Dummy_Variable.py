# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:01:28 2022

@author: Dell
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("C:/360DG/Datasets/animal_category.csv ")
df.shape
df.drop(['Index'],axis=1,inplace=True)
#Check df again
df_new=pd.get_dummies(df)
df_new.shape
# Here we are getting 30 rows and 14 columns
# we are getting two columns for homely and gender ,one column for each is sufficient,let us
#delete second column of gender and second column of homely
df_new.drop(['Gender_Male','Homly_Yes'],axis=1,inplace=True)
df_new.shape
#Now we are getting 30,12 
df_new.rename(columns={'Gender_Female':'Gender','Homly_No':'Homly'},inplace=True)

