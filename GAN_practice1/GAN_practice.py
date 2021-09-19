from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
import numpy as np
import matplotlib.pyplot as plt
import cv2

''''''''''' dataset parsing '''''''''''
#실제 이미지, 라벨 파싱
train_list, test_list = [], []  #txt파일에 저장된 jpg 파일 이름 / 과일 인덱스 리스트로 저장
with open('train.txt') as f:
    for line in f:
        tmp = line.strip().split()
        train_list.append([tmp[0], tmp[1]])  #[0]은 jpg 파일이름, [1]은 과일 인덱스

# with open('test.txt') as f:
#     for line in f:
#         tmp = line.strip().split()
#         test_list.append([tmp[0], tmp[1]])


'''''''''''''image arrangement'''''''''''''
def reading(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  #imread -> 이미지를 다차원 Numpy 배열로 로딩
    img = np.reshape(img, [-1, 10000])  #이미지 100*100
    return img


'''''''''''image, label definition'''''''''''''''
def batch(train_list, batch_size):
    img, paths = [], []
    for i in range(batch_size):
        img.append(reading(train_list[0][0]))
        paths.append(train_list.pop(0))
    return img


###############option################
n_input = 100*100  #입력값의 크기로 이미지의 크기 높이 100 길이 100
n_class = 3  #인덱스의 갯수로 바나나, 오렌지, 사과 3개
n_noise = 128  #생성자의 입력값으로 사용할 노이즈의 크기
total_epoch = 10  #전체 데이터 학습 총 횟수 (세대 학습 숫자)
# batch_size = 1461  #미니배치로 한번에 학습할 데이터의 숫자 1461개씩 학습!
learning_rate = 0.0002
n_hidden = 256  #은닉층 뉴런 개수
#####################################

'''''''''''''generator Model'''''''''''''
generator = Sequential()
generator.add(Dense(128*25*25, input_dim=n_noise, activation=LeakyReLU(0.2)))
#25*25 -> 50*50 ->100*100
generator.add(BatchNormalization())
generator.add(Reshape((25, 25, 128)))
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=5, padding='same'))
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))

'''''''''''''discriminator Model'''''''''''''
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(100, 100, 1), padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable= False

'''''''''''''''connect generator and discriminator/ GAN MODEL'''''''''''''''
ginput = Input(shape=(n_noise,))

dis_output = discriminator(generator(ginput))  #판별자에 생성자에서 만든 이미지 넣어서 판별한 결과
gan = Model(ginput, dis_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan.summary()


'''''''''''''''''''GAN Model trainer'''''''''''''''''''
def gan_train(epoch, batch_size, saving_interval):
    # 훈련 이미지 불러와서 저장!
    x_train = batch(train_list, batch_size)  #x_train 에는 각 이미지 별 1차원으로 된 리스트
                                                      #x_label 에는 각 이미지 인덱스에 맞는 3 -Class의 리스트 존재
    #print(x_train)
    for i in range(len(x_train)):
        #print(x_train[i])
        x_train[i] = x_train[i].astype('float32')
        x_train[i] = (x_train[i]-127.5) / 127.5

    x_train = np.array(x_train)
    x_train = x_train.reshape(len(x_train), 100, 100, 1)
    print("x_train_shape:",x_train.shape)
    #print("float:",  x_train)
    true = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for i in range(epoch):
        #실제 데이터를 판별자에 입력
        idx = np.random.randint(0, len(x_train), batch_size)
        #print("idx:", idx)
        imgs = x_train[idx]
        #print("imgs shape:", imgs.shape)
        #print("imgs:", imgs)
        d_loss_real = discriminator.train_on_batch(imgs, true)

        #가상 이미지를 판별자에 입력
        noise = np.random.normal(0, 1, (batch_size, 128))
        gan_imgs = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(gan_imgs, fake)

        #판별자와 생성자의 오차 계산
        d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
        g_loss = gan.train_on_batch(noise, true)

        print('epoch:%d' % i, 'd_loss:%.4f' %d_loss, 'g_loss:%.4f' %g_loss)

        if i % saving_interval == 0:
            #r,c = 5,5
            noise = np.random.normal(0, 1, (25, 128))
            gan_imgs = generator.predict(noise)

            #rescale images 0-1
            gan_imgs = 0.5*gan_imgs + 0.5

            fig, axs = plt.subplots(5, 5)
            count =0
            for j in range(5):
                for k in range(5):
                    axs[j,k].imshow(gan_imgs[count, :, :, 0], cmap='gray')
                    axs[j,k].axis('off')
                    count+=1
                    fig.savefig("gan_images/gan_img_%d.png"%i)

gan_train(501,32,50)



