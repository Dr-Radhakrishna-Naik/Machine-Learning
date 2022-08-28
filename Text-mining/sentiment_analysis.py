# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 18:03:29 2022

@author: Dell
"""
import pandas as pd
from textblob import TextBlob
sent="This is very nice pendrive"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv("c:/360DG/Datasets/Amazon_reviews.csv")
df.head()
pol=TextBlob(df.loc[0]['review_body']).sentiment.polarity
pol
df['scores']=df['review_body'].apply(lambda x:TextBlob(x).sentiment.polarity)
