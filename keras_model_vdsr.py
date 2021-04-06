"""
    date:       2021/3/30 2:27 下午
    written by: neonleexiang
"""
import os
from keras.models import Model
from keras.layers import Conv2D, Activation, Input, add
# import keras
from keras.optimizers import Adam

from keras import backend as K
import tensorflow as tf


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


class VDSR:
    def __init__(self, image_size, c_dim, is_training, learning_rate=1e-4, batch_size=128, epochs=1500):
        """

        :param image_size:
        :param c_dim:
        :param is_training:
        :param learning_rate:
        :param batch_size:
        :param epochs:
        """
        self.image_size = image_size
        self.c_dim = c_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_training = is_training
        if self.is_training:
            self.model = self.build_model()
        else:
            self.model = self.load()

    def build_model(self):
        """

        :return:
        """
        # model = Sequential()
        #
        # """
        #     这里用了一个基础的知识点是 kernel_initializer    其中：he_normal 为正态分布
        #
        #     He 正态分布初始化器。
        #     它从以 0 为中心，标准差为 stddev = sqrt(2 / fan_in) 的截断正态分布中抽取样本，
        #     其中 fan_in 是权值张量中的输入单位的数量
        # """
        #
        # # 5
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal',
        #                  input_shape=(self.image_size, self.image_size, self.c_dim)))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        #
        # # 10
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        #
        # # 15
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        #
        # # 20
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal'))
        # model.add(Activation('relu'))
        # model.add(Conv2D(self.c_dim, 3, padding='same', kernel_initializer='he_normal'))
        # optimizer = Adam(lr=self.learning_rate)
        # model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[PSNR, 'accuracy'])
        # return model
        """
            因为原来的论文我们是需要实现 残差学习 的， 所以使用 Sequential 的方式暂时是不知道怎么去实现的
            所以参考 [GeorgeSeif] 的 VDSR-Keras 实现方法去尝试 Keras 比较有特点的方式
        """
        input_img = Input(shape=(self.image_size, self.image_size, self.c_dim))

        # 5
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(input_img)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)

        # 10
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)

        # 15
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)

        # 20
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(self.c_dim, 3, padding='same', kernel_initializer='he_normal')(x)
        res_img = x

        # output_img = keras.layers.add([res_img, input_img])
        output_img = add([res_img, input_img])

        model = Model(input_img, output_img)
        optimizer = Adam(lr=0.00001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[PSNR, 'accuracy'])

        return model

    def train(self, X_train, Y_train):
        """

        :param X_train:
        :param Y_train:
        :return:
        """
        history = self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs,
                                 verbose=1, validation_split=0.1)
        if self.is_training:
            self.save()
        return history

    def process(self, inputs):
        """
        predict

        :param inputs:
        :return:
        """
        predicted = self.model.predict(inputs)
        return predicted

    def load(self):
        """
        load data
        :return:
        """
        weight_filename = 'srcnn_weight.hdf5'
        model = self.build_model()
        model.load_weights(os.path.join('./model/', weight_filename))
        return model

    def save(self):
        """
        save data
        :return:
        """
        json_string = self.model.to_json()
        if not os.path.exists('model'):
            os.mkdir('model')
        open(os.path.join('model/', 'srcnn_model.json'), 'w').write(json_string)
        self.model.save_weights(os.path.join('model/', 'srcnn_weight.hdf5'))
        return json_string






