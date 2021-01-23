from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import Embedding, Dense, Flatten, GlobalMaxPooling1D
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

'''''''''''seed setting'''''''''''
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

'''''''''''''read dataset'''''''''''''
news = pd.read_csv('Data_Train.csv', encoding='latin1')
#print(news)  #헤더까지 불러옴 7628*2
news = news.values
# print("news Shape: ", news.shape)

'''''''''''''data processing'''''''''''''''
#토큰화 하기
token = Tokenizer()
token.fit_on_texts(news[:, 0])
#print(token.word_index)
newsVal = news[:, 0]
# print("newsValLen: ", len(newsVal))

sequenceWord = []
for i in range(7628):
    sequenceWord.append(text_to_word_sequence(newsVal[i]))
print(len(sequenceWord))
print(sequenceWord[0])

#불용어(조사) 제거하기
stop_words = set(stopwords.words('english'))

for i in range(7628):
    result = []
    for w in sequenceWord[i]:
        if w not in stop_words:
            result.append(w)
    sequenceWord[i] = result

X = token.texts_to_sequences(sequenceWord)  #토큰으로 지정된 인덱스로 새로운 배열 생성
print(X)
Y = news[:, 1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
print(len(x_train))  #5339
print(len(x_test))  #2289

#가장 길이가 긴 뉴스 찾기
max = 0
for i in range(len(x_train)):
    if max<len(x_train[i]):
        max=len(x_train[i])
print(max)  #593

max = 0
for i in range(len(x_test)):
    if max<len(x_test[i]):
        max=len(x_test[i])
print(max)  #513

#패딩, 서로 다른 길이의 데이터를 500으로 맞춤
x_train = pad_sequences(x_train, maxlen=500)
x_test = pad_sequences(x_test, maxlen=500)
#데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

word_size = len(token.word_index)+1

# '''NN Model'''
# model = Sequential()
# model.add(Embedding(word_size, 200, input_length=500))
# model.add(Flatten())
# model.add(Dense(4, activation='softmax'))

# '''CNN Model'''
# model = Sequential()
# model.add(Embedding(word_size, 200, input_length=500))
# model.add(Dropout(0.3))
# model.add(Conv1D(200, kernel_size=3, padding='valid', activation='relu', strides=1))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(4, activation='softmax'))

# '''LSTM Model'''
# model = Sequential()
# model.add(Embedding(word_size, 200, input_length=500))
# model.add(LSTM(500, activation='tanh'))
# model.add(Dense(4, activation='softmax'))

'''LSTM + CNN Model'''
model = Sequential()
model.add(Embedding(word_size, 200, input_length=500))
model.add(Dropout(0.5))
model.add(Conv1D(200, 3, padding='valid', activation='relu', strides=1))
model.add(LSTM(55))
model.add(Dense(4))
model.add(Activation('sigmoid'))

'''model compile & fit'''
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test, y_test))
print("\n Test accuracy:%.4f"% (model.evaluate(x_test, y_test)[1]))


'''graph로 표현'''
y_vloss = history.history['val_loss']  #테스트셋 오차
y_loss = history.history['loss']  #학습셋 오차
y_vacc = history.history['val_accuracy']  #테스트셋 정확률
y_acc = history.history['accuracy']  #학습셋 정확률

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_acc, marker='.', c='red', label='Trainset_acc')
plt.plot(x_len, y_vacc, marker='.', c='lightcoral', label='Testset_acc')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')
plt.plot(x_len, y_vloss, marker='.', c='cornflowerblue', label='Testset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()

