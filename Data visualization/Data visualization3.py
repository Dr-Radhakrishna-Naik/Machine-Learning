# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:38:27 2022

@author: Dell
"""

import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

lst=[34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
student=pd.DataFrame(lst)
student_series=pd.DataFrame(pd.Series(lst))
student_series.columns=["score"]

student.describe()
##Score in test
# mean is 41.00
#std.deviation=5.05
#median=40
#min=34 and Q1=38.25 ,Q1-min=38-34=4
#max=56 and Q3=41.75 max-Q3=56-41.75= ~22
# means score is right skewed
plt.hist(student_series['score'])
# score is right skewed
plt.boxplot(student_series.score)
#There are outliers in score