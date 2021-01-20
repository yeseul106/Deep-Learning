from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

''''''''''''' read csv file '''''''''''''''
trainData = pd.read_csv('train_house.csv')
testData = pd.read_csv('test_house.csv')

trainData.set_index('Id', inplace=True)
testData.set_index('Id', inplace=True)
test_id_index = testData.index

'''''''feature selection - corrleation'''''''

corrleation = trainData.corr()
corr_features = corrleation.index[abs(corrleation["SalePrice"]) >= 0.3]
print(corr_features)

''''''''''''' divide answer(= sale price) label '''''''''''''''
trainLabel = trainData['SalePrice']
trainData.drop(['SalePrice'], axis=1, inplace=True)

''''''''''''' merge train data + test data '''''''''''''''
trainTestData = pd.concat((trainData, testData), axis=0)
trainTestData_index = trainTestData.index

''''''''''''' drop NA column if (NA >= (total * 0.2)) '''''''''''''''
NA_Ratio = 0.8 * len(trainTestData)
trainTestData.dropna(axis=1, thresh=NA_Ratio, inplace=True)

''''''''''''' divide object type / integer type '''''''''''''''
trainTestData_obj = trainTestData.select_dtypes(include='object')
trainTestData_num = trainTestData.select_dtypes(exclude='object')

''''''''''''' give number to object type '''''''''''''''
for objList in trainTestData_obj:
    trainTestData_obj[objList] = LabelEncoder().fit_transform(trainTestData_obj[objList].astype(str))

#trainTestData_dummy = pd.get_dummies(trainTestData_obj, drop_first=True)
#trainTestData_dummy.index = trainTestData_index

''''''''''''' give mean value to NA '''''''''''''''
imputer = SimpleImputer(strategy='mean')
imputer.fit(trainTestData_num)
trainTestData_impute = imputer.transform(trainTestData_num)
trainTestData_num = pd.DataFrame(trainTestData_impute,
                                columns=trainTestData_num.columns,
                                index=trainTestData_index)

''''''''''''' merge object type + integer type '''''''''''''''
trainTestData = pd.merge(trainTestData_obj, trainTestData_num, left_index=True, right_index=True)

''''''''''''' divide train data / test data '''''''''''''''
trainData = trainTestData[:len(trainData)]
testData = trainTestData[len(trainData):]

''''''''''''' add answer(= sale price) label '''''''''''''''
trainData['SalePrice'] = trainLabel

''''''''''''' check shape '''''''''''''''
#print(trainData.head())        #  -> 1460(row) X 246(col, with answer label)
#print(testData.head())         #  -> 1459(row) X 245(col)
#print(trainTestData.head())    #  -> 2919(row) x 245(col, without answer label)

''''''''''''' check file '''''''''''''''
#trainData.to_csv("check_file_trainData.csv")           #   -> train data file
#testData.to_csv("check_file_testData.csv")             #   -> test data file
#trainTestData.to_csv("check_file_trainTestData.csv")   #   -> train + test data file

seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

print(trainData.info())

dataset = trainData.values
X = dataset[:, 0:74]
Y = dataset[:, 74]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim=74, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100, batch_size=10)

Y_prediction = model.predict(X_test).flatten()

for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))

Y_test_prediction = model.predict(testData)
ID_predict = pd.DataFrame()
ID_predict['Id'] = test_id_index
ID_predict['SalePrice'] = Y_test_prediction
ID_predict.to_csv('submission.csv', index=False)
