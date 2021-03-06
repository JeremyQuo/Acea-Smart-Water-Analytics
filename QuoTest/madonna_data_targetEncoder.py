import sys
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import category_encoders as encoders
import datetime
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler

def get_current_week_monday(temp_time):
    weekday = temp_time.weekday()
    delta=weekday
    monday=temp_time-datetime.timedelta(days=delta)

    return monday
interval='week'
madonna_data=pd.read_csv("madonna_data.csv", encoding='utf-8', sep=',')
for i in range(0, len(madonna_data['Date'])):
    time1 = datetime.datetime.strptime(madonna_data['Date'][i], '%Y-%m-%d')
    if interval=='month':
        madonna_data['Date'][i] = time1.strftime('%Y-%m')
    elif interval=='week':
        time1 = get_current_week_monday(time1)
        madonna_data['Date'][i] = time1.strftime('%Y-%m-%d')
print(madonna_data.columns)

madonna_data.drop(['Unnamed: 0'], axis=1, inplace=True)
madonna_data=madonna_data.sample(frac=1).reset_index(drop=True)

divideRatio=0.9
result=madonna_data['Flow_Rate_Madonna_di_Canneto']
dividingInt = int(madonna_data.shape[0]*divideRatio)
madonna_data.drop(['Flow_Rate_Madonna_di_Canneto'], axis=1, inplace=True)
ss = StandardScaler()
temp=madonna_data[['Rainfall_Settefrati','Temperature_Settefrati']]

temp=ss.fit_transform(temp)
madonna_data[['Rainfall_Settefrati','Temperature_Settefrati']]=temp
train_data = madonna_data[:dividingInt]
test_data = madonna_data[dividingInt:]
train_y=result[:dividingInt]
test_y=result[dividingInt:]

enc = encoders.TargetEncoder()
train_data['Date']=enc.fit_transform(train_data['Date'],train_y)


alphas = 10**np.linspace(5, -2, 100)*0.5
model = linear_model.LassoCV(alphas=alphas, fit_intercept=True,normalize=True)
model.fit(train_data, train_y)

test_data['Date']=enc.transform(test_data['Date'])
pr_y=model.predict(test_data)
print(sm.mean_squared_error(pr_y,test_y))
print(sm.r2_score(pr_y,test_y))



