import sys
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model

import numpy as np
import category_encoders as encoders
from datetime import datetime
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler

lupa_data=pd.read_csv("lupa_data.csv", encoding='utf-8', sep=',')
for i in range(0, len(lupa_data['Date'])):
    time1 = datetime.strptime(lupa_data['Date'][i], '%Y-%m-%d')
    lupa_data['Date'][i] = time1.strftime('%Y-%m')
print(lupa_data.columns)

lupa_data.drop(['Unnamed: 0'], axis=1, inplace=True)
lupa_data=lupa_data.sample(frac=1).reset_index(drop=True)
enc = encoders.OneHotEncoder()
lupa_data=enc.fit_transform(lupa_data)

lupa_data=lupa_data.sample(frac=1).reset_index(drop=True)

ss = StandardScaler()
lupa_data['Rainfall_Terni']=ss.fit_transform(lupa_data['Rainfall_Terni'].values.reshape(-1,1))
divideRatio=0.9
result=lupa_data['Flow_Rate_Lupa']
dividingInt = int(lupa_data.shape[0]*divideRatio)
lupa_data.drop(['Flow_Rate_Lupa'], axis=1, inplace=True)
train_data = lupa_data[:dividingInt]
test_data = lupa_data[dividingInt:]
train_y=result[:dividingInt]
test_y=result[dividingInt:]






alphas = 10**np.linspace(5, -2, 100)*0.5
model = linear_model.LassoCV(alphas=alphas, fit_intercept=True,normalize=True)
model.fit(train_data, train_y)

pr_y=model.predict(test_data)
print(sm.mean_squared_error(pr_y,test_y))
print(sm.r2_score(pr_y,test_y))



