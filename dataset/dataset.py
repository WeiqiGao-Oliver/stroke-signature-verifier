from torch.utils import data
import torch
import cv2
import numpy as np

class dataset(data.Dataset):
    def __init__(self, root='/mnt/IDN/CEDAR/', train=True):  # 初始化数据集位置
        super(dataset, self).__init__()
        if train:  # 设置训练路径
            path = root + 'gray_train.txt'
        else:      # 设置测试路径
            path = root + 'gray_test.txt'
        # 逐行读取样本数据
        with open(path, 'r') as f:
            lines = f.readlines()
        # 设置标签数据
        self.labels = []
        # 设置样本数据
        self.datas = []
        # 针对每组样本进行处理
        for line in lines:
            # 将每行样本名称划分为参考图像、测试图像和标签三种路径形式
            refer, test, label = line.split()
            #print(root + refer)
            # 读取参考图像文件二进制形式
            refer_img = cv2.imread(root + refer, 0)
            # 读取测试图像文件二进制形式
            test_img = cv2.imread(root + test, 0)
            # 读取参考图像文件二进制灰度图形式
            refer_img = refer_img.reshape(-1, refer_img.shape[0], refer_img.shape[1])
            # 读取测试图像文件二进制灰度图形式
            test_img = test_img.reshape(-1, test_img.shape[0], test_img.shape[1])
            # 最终级联参考图像和测试图像
            refer_test = np.concatenate((refer_img, test_img), axis=0)
            self.datas.append(refer_test)
            self.labels.append(int(label))

    def __len__(self):
        # 设置长度函数
        return len(self.labels)

    def __getitem__(self, index):
        # 设置逐个读取图像
        return torch.FloatTensor(self.datas[index]), float(self.labels[index])

# img = cv2.imread('dataset/original_2_9.png')
# print(img.shape)
