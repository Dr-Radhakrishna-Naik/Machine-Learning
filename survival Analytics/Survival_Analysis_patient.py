"""
The following dataset contains patient ID, follow up, event type,
 and scenarios. Build a survival analysis model on the given data.
 1.	Business Problem
1.1.	What is the business objective?
       Typically in clinical trials patients are not taking treatment 
       at the same time,but it carried over a period of months.
       Patients are followed up until either the study is completed or they die.
1.2.	Are there any constraints?
        Although actual survival time is observed for number of patients,
        after some time patient may lost life or discontinue followup.
        which may cause loss in data in left hand side or right hand side.
 
"""
import lifelines

import pandas as pd
# Loading the the survival un-employment data
survival_patient = pd.read_csv("C:/360DG/Datasets/Patient.csv")
survival_patient.head()
survival_patient.describe()

survival_patient["Followup"].describe()

# Spell is referring to time 
T = survival_patient.Followup

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=survival_patient.Eventtype)
survival_patient['Scenario']=survival_patient['Scenario'].apply(lambda x:1)
# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group, here group is ui
survival_patient.Scenario.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[survival_patient.Scenario==1], survival_patient.Eventtype[survival_patient.Scenario==1], label='1')
ax = kmf.plot()
#The benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
#one can easily predict survival rate of patient after taking tratment over a period of time