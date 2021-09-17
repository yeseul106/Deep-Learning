import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''''''''''''''read csv file'''''''''''''''
trainData = pd.read_csv('Data/Taxi_train.csv',parse_dates = ["pickup_datetime" , "dropoff_datetime"])
testData =  pd.read_csv('Data/Taxi_test.csv',parse_dates = ["pickup_datetime"])
print("train 데이터 shape :", trainData.shape)
trainData.head(5)

print("test 데이터 shape :", testData.shape)
testData.head(5)

train_dropoff_datetime = trainData['dropoff_datetime']
train_trip_duration = trainData['trip_duration']
trainData.drop(['dropoff_datetime','trip_duration'],axis=1, inplace=True)
trainData.columns

'''''''''''train 데이터셋와 test 데이터셋 합치기'''''''''''
trainTestData = pd.concat((trainData, testData), axis=0)
print(trainData.shape[0] + testData.shape[0])
print(trainTestData.shape)

'''''''''''''날짜 변수(시계열 데이터) 세분화 하기'''''''''''''
trainTestData['pickup_datetime'].head(3)
trainTestData['pickup_year'] = trainTestData['pickup_datetime'].dt.year
trainTestData['pickup_month'] = trainTestData['pickup_datetime'].dt.month
trainTestData['pickup_day'] = trainTestData['pickup_datetime'].dt.day
trainTestData['pickup_hour'] = trainTestData['pickup_datetime'].dt.hour
trainTestData['pickup_minute'] = trainTestData['pickup_datetime'].dt.minute
trainTestData['pickup_second'] = trainTestData['pickup_datetime'].dt.second
trainTestData["pickup_dayofweek"] = trainTestData["pickup_datetime"].dt.dayofweek

'''''''''''''object 타입 - 숫자 형태로 바꿔주기'''''''''''''
from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(trainTestData['store_and_fwd_flag'])
trainTestData['store_and_fwd_flag'] = e.transform(trainTestData['store_and_fwd_flag'])
trainTestData['store_and_fwd_flag'].value_counts() # 0과 1로 잘 바뀌었는지 확인

''''''''''''' divide train data / test data '''''''''''''''
trainData = trainTestData[:len(trainData)]
testData = trainTestData[len(trainData):]
print(trainData.shape)
print(testData.shape)

'''''''''''''train 데이터셋에 원래 제거했던 열 추가해주기'''''''''''''
trainData['trip_duration'] = train_trip_duration
trainData.head(5)

'''''''''''''0인 값 제거하기'''''''''''''
trainData[trainData.passenger_count == 0].shape # (60, 17)
trainData = trainData[trainData.passenger_count != 0]
trainData.shape

'''''''''''''''haversine 공식으로 거리 구하기'''''''''''''''
from math import sin, cos, sqrt, atan2, radians # approximate radius of earth in km
import numpy as np

def haversine(df):
    lon1 = df['pickup_longitude']
    lat1 = df['pickup_latitude']
    lon2 = df['dropoff_longitude']
    lat2 = df['dropoff_latitude']
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * np.arcsin(sqrt(a))
    km = 6367 * c
    return km

# 'distance' 열로 추가해주기
trainData['distance'] = trainData.apply(lambda trainData: haversine(trainData), axis=1)
testData['distance'] = testData.apply(lambda testData: haversine(testData), axis=1)

'''''''''''''''feature 고르기'''''''''''''''
features = [ 'vendor_id',
                 'passenger_count', 'pickup_longitude', 'pickup_latitude',
                 'dropoff_longitude', 'dropoff_latitude',
                 'pickup_year', 'pickup_month', 'pickup_day', 'pickup_hour',
                 'pickup_minute', 'pickup_second', 'pickup_dayofweek',
                 'store_and_fwd_flag','distance' ]

'''''''''''''데이터 feature 바꾸기'''''''''''''
X = trainData[features]
testData = testData[features]
Y = trainData['trip_duration']
print(X.shape)
X.head(5)

Y = np.log(Y+1)
print(Y.shape)
Y.head()

# it will show better score

'''''''''''''keras NN 모델 구현하기'''''''''''''
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim=15, activation='relu', kernel_initializer = 'normal'))
model.add(Dropout(0.6))

model.add(Dense(10, activation='relu', kernel_initializer = 'normal'))

model.add(Dropout(0.3))

model.add(Dense(1))

model.compile(loss = 'mean_squared_logarithmic_error', optimizer ="adam")
print(X.shape)
print(Y.shape)
model.fit(X, Y, batch_size=32, epochs=10)

Y_test_prediction = model.predict(testData, batch_size=32)

predictions = np.exp(Y_test_prediction) - 1
print(predictions.shape)
print(predictions[0:10])

submission = pd.read_csv("Data/sample_submission.csv")
submission["trip_duration"] = predictions

submission.to_csv("submission.csv", index=False)
