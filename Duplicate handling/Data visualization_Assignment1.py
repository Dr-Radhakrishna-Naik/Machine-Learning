# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 16:06:20 2022

@author: Dell
"""

import pandas as pd
retail=pd.read_csv("C:/360DG/Datasets/OnlineRetail.csv",encoding='latin1')
retail.dtypes
retail.UnitPrice=retail.UnitPrice.astype("int64")
retail.dtypes
duplicate=retail.duplicated()
sum(duplicate)
retail1=retail.drop_duplicates()
retail
# it has shown 541909 rows and 8 columns
retail1
#it has shown 536625 rows and 8 columns
retail1.describe()
#Here Q1 for quantity=1 and Q3=10 L1=9,however the upper limit=80995-10=80885
# it means it is right skewed
#retail1['quantity_new']=pd.cut(retail1['Quantity'],bins=[min(retail1.Quantity)-1,retail1.Quantity.mean(),max(retail1.Quantity)],labels=["low","high"])
#retail1.quantity_new.value_counts()
import matplotlib.pyplot as plt
import numpy as np
retail1.Description=retail1.Description.astype(str)
retail1.dtypes
plt.bar(retail1.Description,retail1.Quantity, color ='maroon')
