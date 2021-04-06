"""
    date:       2021/3/9 4:12 下午
    written by: neonleexiang
"""
import numpy as np
import os
import random
import shutil
import cv2 as cv
import tensorflow as tf
import self_load_data

"""
    Tensorflow 需要写 data_loader method 去返回数据供给给模型进行训练
"""

"""
    this python file is to set the data preprocessing
    for tensorflow 2.0, we should use the data_loader to read our data into model
    for img we should remember to divide its pixel into 255
    also we should reset our img size into 128*128
"""


def img_process_train(img):
    """
        resize it into 32*32 then resize it into 128*128 by using inter_cubic
        according to the paper we use bicubic methods to resize the img into the
        same size with High Resolution Image, then training the CNN model and output
        the Super Resolution image.
    :param img:
    :return:
    """
    train = cv.resize(img, (32, 32), interpolation=cv.INTER_NEAREST)
    train = cv.resize(train, (128, 128), interpolation=cv.INTER_CUBIC)
    return np.array(train).reshape((128, 128, 1)) / 255.


def img_process_label(img):
    return np.array(cv.resize(img, (128, 128), interpolation=cv.INTER_CUBIC)).reshape((128, 128, 1)) / 255.


def img_data_list(path, tag='label'):
    if tag == 'train':
        return [img_process_train(
            cv.imread(os.path.join(path, img_path), cv.IMREAD_GRAYSCALE)) for img_path in os.listdir(path)]
    else:
        return [img_process_label(
            cv.imread(os.path.join(path, img_path), cv.IMREAD_GRAYSCALE)) for img_path in os.listdir(path)]


def data_process(train_path='datasets/train/', test_path='datasets/test/'):
    # split the data into train data and label and test and so on
    # train_data = np.array(img_data_list(train_path, 'train'))
    # train_label = np.array(img_data_list(train_path))
    # test_data = np.array(img_data_list(test_path, 'train'))
    # test_label = np.array(img_data_list(test_path))

    # cifar10

    # load data
    print('loading_data...')
    (train_images, train_labels), (test_images, test_labels) = self_load_data.load_data('cifar-10-python.tar')
    # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # print(train_images[0].shape)
    print('data_preprocessing')
    train_data = np.array([img_process_train(cv.cvtColor(img, cv.COLOR_RGB2GRAY)) for img in train_images[:10]])
    print(train_data.shape)
    train_label = np.array([img_process_label(cv.cvtColor(img, cv.COLOR_RGB2GRAY)) for img in train_images[:10]])
    test_data = np.array([img_process_train(cv.cvtColor(img, cv.COLOR_RGB2GRAY)) for img in test_images[:12]])
    test_label = np.array([img_process_label(cv.cvtColor(img, cv.COLOR_RGB2GRAY)) for img in test_images[:12]])

    return train_data, train_label, test_data, test_label


class SRCNNLoader:
    def __init__(self):
        """
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
        """

        self.train_data, self.train_label, self.test_data, self.test_label = data_process()
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

        """
        # # to test the data_loader
        # print(self.num_train_data, self.num_test_data)
        # cv.imshow('train_data', self.train_data[0])
        # cv.imshow('train_data_label', self.train_label[0])
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        """

    # 可以用 get_batch 方法去返回对应的数据喂入模型进行训练
    # 这里我们实现是通过random的方式返回随机的数据进行训练。
    def get_batch(self, batch_size):
        # # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]


def get_data(data_path='Train'):
    """
        using random method to split the data into train data and test data
    :param data_path:
    :return:
    """
    if not os.path.exists('datasets'):
        os.mkdir('datasets/')
        os.mkdir('datasets/train/')
        os.mkdir('datasets/test/')
        lst_dir = os.listdir(data_path)
        test_img_path = random.sample(lst_dir, int(0.1*len(lst_dir)))
        for img in lst_dir:
            if img in test_img_path:
                shutil.copy(os.path.join(data_path, img), os.path.join('datasets/test/', img))
            else:
                shutil.copy(os.path.join(data_path, img), os.path.join('datasets/train/', img))


if __name__ == '__main__':
    # first we should use get_data() method to split our img data

    # get_data()

    data_loader = SRCNNLoader()
    # exit(0)



