
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''''''''''''''read csv file'''''''''''''''
trainData = pd.read_csv('C:/Users/82109/Documents/Deep Learning/kaggle_Blue_Book_for_Bulldozers/Train.csv',parse_dates = ["saledate"])
validData =  pd.read_csv('C:/Users/82109/Documents/Deep Learning/kaggle_Blue_Book_for_Bulldozers/Valid.csv',parse_dates = ["saledate"])
testData = pd.read_csv('C:/Users/82109/Documents/Deep Learning/kaggle_Blue_Book_for_Bulldozers/Test.csv',parse_dates = ["saledate"])
print("train 데이터 shape :",trainData.shape)

print("Valid 데이터 shape :",validData.shape)

print(trainData.shape)
print(validData.shape)
print(testData.shape)

print(trainData.shape)


'''''''''''feature 중 NA 값 있는지 확인'''''''''''

# 1) 결측치 열 제거 해보기

df = pd.DataFrame(trainData.isna().sum() >= 100000)
df_index = df.index[df[0]==True].tolist()
print(trainData[df_index].columns)


df = pd.DataFrame(trainData.isna().sum() >= 100000)
df_index = df.index[df[0]==True].tolist()
trainData = trainData.drop(columns=df_index)


# validData와 testData 역시 feature 제거해주기
validData = validData.drop(columns=df_index)
testData = testData.drop(columns=df_index)


print(trainData.shape)
print(validData.shape)
print(testData.shape)



##null 값이 존재하는 feature


#  Machine_Appendix.csv 파일 이용하기

machine_appendix = pd.read_csv('C:/Users/82109/Documents/Deep Learning/kaggle_Blue_Book_for_Bulldozers/Machine_Appendix.csv')


# 이 파일도 역시나 너무 많은 결측치가 있는 열을 삭제

machine_df = pd.DataFrame(machine_appendix.isna().sum() >= 100000)
machine_df_index = machine_df.index[machine_df[0]==True].tolist()
machine_appendix = machine_appendix.drop(columns=machine_df_index)


print(trainData.shape)
print(machine_appendix)

## Machine_appendix 파일을 valid와 test 데이터셋에도 사용할 수 있는 것인가?


# valid, test 모든 행이 Machine_appendix에 있는지 확인
# valid 모든 행의 MachineID가 Machine_appendix에 포함되어있는지 확인
Valid_machineID = set(validData['MachineID'])
Machine_appendix_machineID = set(machine_appendix['MachineID'])
cross_valid = Machine_appendix_machineID & Valid_machineID
print(len(cross_valid))
print(len(Valid_machineID)) #valid 데이터는 다 들어가있는 것을 확인

Test_machineID = set(testData['MachineID'])
cross_test = Machine_appendix_machineID & Test_machineID
print(len(cross_test))
print(len(Test_machineID)) #test 데이터 역시 다 들어가있는 것을 확인

#machine_appendix에서 trainData와 겹치는 열을 제거하기 => testData, validData 둘다 같음
machine_appendix_columns = machine_appendix.columns.values.tolist()
trainData_columns = trainData.columns.values.tolist()
unique_columns = list(set(machine_appendix_columns) - set(trainData_columns))
print(unique_columns)
drop_colunms = list(set(machine_appendix_columns)-set(unique_columns))
print(drop_colunms)

#MachineID 먼저 빼놓기
machine_appendix_machinID = machine_appendix['MachineID']
machine_appendix = machine_appendix.drop(columns=drop_colunms,axis=1)

# 고유 key인 MachineID는 다시 붙여주기
machine_appendix['MachineID'] = machine_appendix_machinID

#두 데이터 프레임 합치기
total_trainData = pd.merge(machine_appendix, trainData , how='inner', on='MachineID')
print(trainData.shape)
print(machine_appendix.shape)
print(total_trainData.shape)

print(validData.shape)
validData = pd.merge(machine_appendix, validData , how='inner', on='MachineID')
print(machine_appendix.shape)
print(validData.shape)
#validData.head(5)

print(testData.shape)
testData = pd.merge(machine_appendix, testData , how='inner', on='MachineID')
print(machine_appendix.shape)
print(testData.shape)

#trainData 의 정답 label 값 따로 저장
train_label = total_trainData['SalePrice']
total_trainData.drop(['SalePrice'],axis=1,inplace=True)

trainData = total_trainData
print(trainData.shape)

# 1-1) Null 값이 있는 행 모두 제거해주기

print('null 값 있는 행 제거 전 trainData shape : ', trainData.shape)
print('null 값 있는 행 제거 전 testData shape : ', testData.shape)
print('null 값 있는 행 제거 전 validData shape : ', validData.shape)

# null값을 모두 제거해주기
trainData = trainData.dropna()
testData = testData.dropna()
validData = validData.dropna()


print('제거 후 trainData shape : ', trainData.shape)
print('제거 후 testData shape : ', testData.shape)
print('제거 후 validData shape : ', validData.shape)


#상관계수 파악을 위해 trainData 복사해놓기
corr_trainData = trainData
print(corr_trainData.shape)

#trainData 와  validData, testData 모두 합쳐놓기 일단 ! (어짜피 처리를 동일하게 해주어야 하기 때문이다)
All_Data = pd.concat((trainData,testData,validData),axis=0)
print(trainData.shape[0]+validData.shape[0]+testData.shape[0])
print(All_Data.shape)

##object 타입 one-hot encoding

print(All_Data.shape)


#날짜 변수 (시계열 데이터) 세분화 하기
All_Data['saledate_year'] = All_Data['saledate'].dt.year
All_Data['saledate_month'] = All_Data['saledate'].dt.month
All_Data['saledate_day'] = All_Data['saledate'].dt.day
All_Data.drop(['saledate'],axis=1,inplace=True)


print(len(All_Data['fiManufacturerDesc'].unique()))
print(len(All_Data['fiManufacturerID'].unique()))

print(len(All_Data['PrimarySizeBasis'].unique()))
print(len(All_Data['PrimaryLower'].unique()))
print(len(All_Data['PrimaryUpper'].unique()))

print('\n')
print(len(All_Data['fiModelDesc'].unique()))
print(len(All_Data['fiBaseModel'].unique()))
print(len(All_Data['fiProductClassDesc'].unique()))
print(len(All_Data['state'].unique()))
print(len(All_Data['ProductGroup'].unique()))
print(len(All_Data['ProductGroupDesc'].unique()))
print(len(All_Data['Enclosure'].unique()))
print(len(All_Data['Hydraulics'].unique()))

#corr_trainData는 숫자 형태로 라벨링 해주기 !
#그 전에 필요 없는 feature부터 제거
drop_objectFeature = ['fiManufacturerDesc', 'fiModelDesc' ,'fiBaseModel', 'fiProductClassDesc', 'ProductGroupDesc', 'state']
corr_trainData = corr_trainData.drop(columns=drop_objectFeature)
print(corr_trainData.shape)
corr_trainData.head(5)

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
encoding_list = ['PrimarySizeBasis','ProductGroup','Enclosure','Hydraulics']

for i in encoding_list:
  e.fit(corr_trainData[i])
  corr_trainData[i] = e.transform(corr_trainData[i])

corr_trainData.head(5)

'''''''''''''train 데이터셋에 원래 제거했던 열 추가해주기'''''''''''''
corr_trainData['SalePrice'] = train_label

'''''''상관계수 확인해보기'''''''
corrleation = corr_trainData.corr()
corr_features = corrleation.index[abs(corrleation["SalePrice"]) >= 0.3]
print(corr_features)

# 시각적으로 보기 위해 표를 이용하여 확인
import matplotlib.pyplot as plt
import seaborn as sns

#그래프 크기 결정
plt.figure(figsize=(12,12))
sns.heatmap(corrleation,linewidths=0.1,vmax=0.5,cmap=plt.cm.gist_heat,linecolor='white',annot=True)
plt.show()

# '''''''''''''object 타입 - 숫자 형태로 바꿔주기'''''''''''''
# e = LabelEncoder()
# e.fit(All_Data['fiModelDesc'])
# All_Data['fiModelDesc'] = e.transform(All_Data['fiModelDesc'])
# All_Data['fiModelDesc'].value_counts() # 0과 1로 잘 바뀌었는지 확인
# # print(len(All_Data['fiManufacturerDesc'].unique()))

# e.fit(All_Data['fiBaseModel'])
# All_Data['fiBaseModel'] = e.transform(All_Data['fiBaseModel'])
# All_Data['fiBaseModel'].value_counts() # 0과 1로 잘 바뀌었는지 확인

# 중요!!!!!!!!!!!!1 LabelEncoder은 트리 형태의 머신 러닝 모델에서는 상관이 없지만 선형회귀 같은 경우에는 좋지 않은 방법 ! (숫자의 크기 특성이 반영되기 때문에)

# 다시 점검하기 !! 모든 object 인코딩 하는가??

All_Data = pd.get_dummies(All_Data, columns=['PrimarySizeBasis','ProductGroup','Enclosure','Hydraulics'], drop_first=True)

All_Data.info()
All_Data.head(5)

drop_objectFeature = ['fiManufacturerDesc', 'fiModelDesc' ,'fiBaseModel', 'fiProductClassDesc', 'ProductGroupDesc', 'state']
All_Data = All_Data.drop(columns=drop_objectFeature)
print(All_Data.shape)
All_Data.head(5)

# 검증용 데이터 정답 레이블
ValidSolution = pd.read_csv('C:/Users/82109/Documents/Deep Learning/kaggle_Blue_Book_for_Bulldozers/ValidSolution.csv')
Y_label = ValidSolution['SalePrice']

''''''''''''' divide train data / test data '''''''''''''''
trainData = All_Data[:len(trainData)]
testData = All_Data[len(trainData):len(trainData)+len(testData)]
validData = All_Data[len(trainData)+len(testData):]
print(trainData.shape)
print(testData.shape)
print(validData.shape)


'''''''''''''train 데이터셋에 원래 제거했던 열 추가해주기'''''''''''''
trainData['SalePrice'] = train_label

print(trainData.shape)
print(testData.shape)
print(validData.shape)

# 검증용 데이터 정답 레이블 붙이기
ValidSolution.drop(['Usage'],axis=1,inplace=True)
validData = pd.merge(validData, ValidSolution , how='inner', on='SalesID')
print(validData.shape)
print(validData.head(5))
print(trainData.head(5))

#trainData와 validData 합치기
train_valid_Data = pd.concat((trainData, validData), axis=0)
print("합친 후: ", train_valid_Data.shape)
print("",trainData.shape[0] + validData.shape[0])


from sklearn.model_selection import train_test_split
import tensorflow as tf

Y = train_valid_Data['SalePrice']
X = train_valid_Data.drop(['SalePrice','SalesID'],axis=1)

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

print(X.shape)
print(Y.shape)
#학습셋과 테스트셋의 구분
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.3, random_state = seed)



Y = np.log(Y+1)
print(Y.shape)

# it will show better score

'''''''''''''keras NN 모델 구현하기'''''''''''''
from keras.models import Sequential
from keras.layers import Dense, Dropout
import kerasㄴㄴㄴㄴㄴ
import tensorflow as tf
from keras.callbacks import EarlyStopping

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(100, input_dim=33, activation='relu', kernel_initializer = 'normal'))
model.add(Dropout(0.6))
model.add(Dense(60, activation='relu', kernel_initializer = 'normal'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu', kernel_initializer = 'normal'))
model.add(Dropout(0.3))
model.add(Dense(5, activation='relu', kernel_initializer = 'normal'))
model.add(Dropout(0.3))

model.add(Dense(1))


#Adam = keras.optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
model.compile(loss='mean_squared_logarithmic_error', optimizer="adam", metrics=['accuracy'])
print(X.shape)
print(Y.shape)

hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=100, verbose=1)

#그래프 그려보기

# 검증용셋의 오차
y_vloss = hist.history['val_loss']

# 학습셋의 오차
y_loss = hist.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label = "Testset_loss")
plt.plot(x_len, y_loss, marker='.', c="blue", label = "Trainset_loss")

#그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

Y_test = Y_test.reset_index(drop=True)

print(Y_test.head(5))

'''''''valid data로 검증 해보기'''''''
#ValidSolution = pd.read_csv('/Users/82109/Documents/Deep Learning/kaggle_Blue_Book_for_Bulldozers/ValidSolution.csv')
Y_prediction = hist.predict(X_test).flatten()

for i in range(30):
  label = Y_test[i]
  prediction = Y_prediction[i]
  print("실제 가격: {:.3f}, 예상 가격: {:.3f}".format(label, prediction))
