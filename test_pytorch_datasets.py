"""
    date:       2021/3/25 1:32 下午
    written by: neonleexiang
"""
from torch.utils.data import Dataset
import self_load_data
import cv2 as cv
import numpy as np
import torch
from data_preprocessing import img_process_train, img_process_label

# ------> we import the img preprocess method to avoid the duplicated code

# # reconstruct the img_process_methods
# def img_process_train(img):
#     """
#         resize it into 32*32 then resize it into 128*128 by using inter_cubic
#         according to the paper we use bicubic methods to resize the img into the
#         same size with High Resolution Image, then training the CNN model and output
#         the Super Resolution image.
#     :param img:
#     :return:
#     """
#     train = cv.resize(img, (16, 16), interpolation=cv.INTER_NEAREST)
#     train = cv.resize(train, (32, 32), interpolation=cv.INTER_CUBIC)
#     return np.array(train).reshape((32, 32, 1)) / 255.
#
#
# def img_process_label(img):
#     return np.array(img).reshape((32, 32, 1)) / 255.


class TrainDataset(Dataset):
    def __init__(self, train_size=100000):
        super(TrainDataset, self).__init__()
        self.train_size = train_size
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self_load_data.load_data('cifar-10-python.tar')
        self.train_data = np.array([img_process_train(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
                                    for img in self.train_images[:self.train_size]])
        self.train_label = np.array([img_process_label(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
                                     for img in self.train_images[:self.train_size]])

        # we change data before return by using the permute method

        # self.train_data = torch.from_numpy(self.train_data).permute(0, 3, 1, 2).to(torch.float32)
        # self.train_label = torch.from_numpy(self.train_label).permute(0, 3, 1, 2).to(torch.float32)

        self.train_data = torch.from_numpy(self.train_data)
        self.train_label = torch.from_numpy(self.train_label)
        self.len = self.train_size

    # fix the index error cause by the different between [H, W, C] data -> [C, H, W]
    def __getitem__(self, index):
        return self.train_data[index].permute(2, 0, 1).to(torch.float32), \
               self.train_label[index].permute(2, 0, 1).to(torch.float32)

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self, test_size=1000):
        super(TestDataset, self).__init__()
        self.test_size = test_size
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self_load_data.load_data(
            'cifar-10-python.tar')
        # print('data_preprocessing')
        self.test_data = np.array([img_process_train(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
                                   for img in self.test_images[:self.test_size]])
        self.test_label = np.array([img_process_label(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
                                    for img in self.test_images[:self.test_size]])
        # self.test_data = torch.from_numpy(self.test_data).permute(0, 3, 1, 2).to(torch.float32)
        # self.test_label = torch.from_numpy(self.test_label).permute(0, 3, 1, 2).to(torch.float32)
        self.test_data = torch.from_numpy(self.test_data)
        self.test_label = torch.from_numpy(self.test_label)

        self.len = self.test_size

    def __getitem__(self, index):
        return self.test_data[index].permute(2, 0, 1).to(torch.float32), \
               self.test_label[index].permute(2, 0, 1).to(torch.float32)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    test = TrainDataset(10)
    # test_data = test[0][0]
    # print(test_data)
    # test_data_permute = test_data.permute(2, 0, 1)
    # print(test_data_permute)
