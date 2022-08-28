"""
1.	Business Problem
1.1.	What is the business objective?
      Machine learning, in particular, can predict patients’ survival 
      from their data and can individuate the most important features 
      among those included in their medical records.
1.2.	Are there any constraints?

"""
import pandas as pd
# Loading the the survival un-employment data
survival_ECG = pd.read_excel("C:/360DG/Datasets/ECG_Surv.xlsx")
survival_ECG.head()
survival_ECG=survival_ECG.iloc[:,0:4]

survival_ECG.columns="spell","event","age","ui"


survival_ECG.describe()




survival_ECG["spell"].describe()

# Spell is referring to time 
T = survival_ECG.spell

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=survival_ECG.event)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group, here group is ui
survival_ECG.ui.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[survival_ECG.ui==1], survival_ECG.event[survival_ECG.ui==1], label='1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[survival_ECG.ui==0], survival_ECG.event[survival_ECG.ui==0], label='0')
kmf.plot()
#the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
#The use of survival analysis in medical field has the potential to impact on
# clinical practice, becoming a new supporting tool for physicians when predicting if a heart failure patient will survive or not. Indeed, medical doctors aiming at understanding if a patient will survive after heart failure may focus mainly on serum creatinine and ejection fraction.
