"""
    date:       2021/3/30 5:39 下午
    written by: neonleexiang
"""
from keras_model_vdsr import VDSR
import cv2 as cv
import self_load_data
import numpy as np
import tensorflow as tf


print('test tensorflow is built with cuda ------->', tf.test.is_built_with_cuda())
print('test tensorflow is built with gpu ------->', tf.test.is_built_with_gpu_support())
print('test gpu is available ------->', tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


def img_process_train(img):
    """
        resize it into 32*32 then resize it into 128*128 by using inter_cubic
        according to the paper we use bicubic methods to resize the img into the
        same size with High Resolution Image, then training the CNN model and output
        the Super Resolution image.
    :param img:
    :return:
    """
    train = cv.resize(img, (16, 16), interpolation=cv.INTER_NEAREST)
    train = cv.resize(train, (32, 32), interpolation=cv.INTER_CUBIC)
    return np.array(train).reshape((32, 32, 1)) / 255.


def img_process_label(img):
    return np.array(img).reshape((32, 32, 1)) / 255.


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
    train_data = np.array([img_process_train(cv.cvtColor(img, cv.COLOR_RGB2GRAY)) for img in train_images[:50000]])
    print(train_data.shape)
    train_label = np.array([img_process_label(cv.cvtColor(img, cv.COLOR_RGB2GRAY)) for img in train_images[:50000]])
    test_data = np.array([img_process_train(cv.cvtColor(img, cv.COLOR_RGB2GRAY)) for img in test_images[:1000]])
    test_label = np.array([img_process_label(cv.cvtColor(img, cv.COLOR_RGB2GRAY)) for img in test_images[:1000]])

    return train_data, train_label, test_data, test_label

def mse(y, t):
    return np.mean(np.square(y - t))


def psnr(y, t):
    return 20 * np.log10(255) - 10 * np.log10(mse(y, t))


def trains_and_test_by_psnr(image_size, c_dim, is_training, learning_rate, batch_size, epochs):
    vdsr = VDSR(
        image_size=image_size,
        c_dim=c_dim,
        is_training=is_training,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )
    print('-------> data processing')
    X_train, Y_train, X_test, Y_test = data_process()
    vdsr.train(X_train, Y_train)

    print('-------------------> then begin to test')
    vdsr_test = VDSR(
        image_size=image_size,
        c_dim=c_dim,
        is_training=False,
    )

    predicted_list = []
    for img in X_test:
        # print('img shape = ', img.shape)
        predicted = vdsr_test.process(img.reshape(1, img.shape[0], img.shape[1], 1))
        # print('predicted shape = ', predicted.shape)
        predicted_list.append(psnr(predicted.reshape(predicted.shape[1], predicted.shape[2], 1) * 255, img * 255))

    print(np.mean(predicted_list))

    # predicted_list = []
    # for img in X_test:
    #     predicted = srcnn.process(img.reshape(1, img.shape[0], img.shape[1], 1))
    #     predicted_list.append(predicted.reshape(predicted.shape[1], predicted.shape[2], 1))
    # n_img = len(predicted_list)
    # dirname = 'result'
    # for i in range(n_img):
    #     img_name = 'image{:02}'.format(i)
    #     print('saving ------>', img_name)
    #     print(X_test[i].shape)
    #     # cv2.imwrite(os.path.join(dirname, imgname + '_original.bmp'), X_pre_test[i])
    #     cv.imwrite(os.path.join(dirname, img_name + '_input.bmp'), X_test[i] * 255)
    #     cv.imwrite(os.path.join(dirname, img_name + '_answer.bmp'), Y_test[i] * 255)
    #     cv.imwrite(os.path.join(dirname, img_name + '_predicted.bmp'), predicted_list[i] * 255)


if __name__ == '__main__':
    trains_and_test_by_psnr(32, 1, True, 0.00001, 64, 500)




