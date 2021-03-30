"""
    date:       2021/3/30 2:27 下午
    written by: neonleexiang
"""
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.core import Activation


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
        model = Sequential()
        model.
