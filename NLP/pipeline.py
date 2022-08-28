# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:15:04 2022

@author: Dell
"""


#############Building pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
# saving and loading the models
#pip install joblib
import pickle
from joblib import dump,load
from sklearn.feature_extraction.text import TfidfVectorizer
corpus=['Data science is the most demanding job role in the market','It is combination of both maths and business skills at time',
        'Natural language processing is a part of Data Science']
tfidf_model=TfidfVectorizer()
print(tfidf_model.fit_transform(corpus).todense())
tfidf_model_loaded=load('tfidf_model.joblib')

######
text=['Once upon a time there lived a programmer name Sharat.He along with his close friend trained many students']

###picle library
pickle.dump(tfidf_model,open("tfidf_model.pickle.dat","wb"))
loaded_model=pickle.load(open("tfidf_model.pickle.dat","rb"))
print(loaded_model.fit_transform(text).todense())
