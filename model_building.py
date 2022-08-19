import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = pd.read_csv('train_all_feature.csv')
y = pd.read_csv('y_train_new.csv') #0: nonfraudulent provider,1:fraudulent provider
 
top11features= ['TotalTeDiagCode',
 'TotalTeProcCode',
 'MaxHospitalDays',
 'MaxDiagCodeNumPerClaim',
 'MedianHospitalDays',
 'ClmsperBene',
 'uniqBeneCount',
 'MajorRace',
 'MeanLowFreqDiagCodeNumPerClaim',
 'InClmsPct',
 'TotalInscClaimAmtReimbursed']

X_train = X[top11features]
y_train = y.iloc[:,0].ravel() 

steps = [('rescale', StandardScaler()), 
          ('logreg', LogisticRegression(C = 0.1, penalty = 'l2', solver = 'newton-cg', class_weight = 'balanced',random_state=42,max_iter=10000))]

model = Pipeline(steps)
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
