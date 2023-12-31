import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pynvml import *
import os
from tensorboardX import SummaryWriter
import time

from dataset.dataset import dataset
from models.net import net
from loss import loss
os.environ['CUDA_VISIBLE_DEVICES']='0'
# 导入函数库，设置GPU

# 定义准确率计算函数
def compute_accuracy(predicted, labels):
    # 得出三部分预测值对应的结果
    for i in range(3):
        predicted[i][predicted[i] > 0.5] = 1
        predicted[i][predicted[i] <= 0.5] = 0
    predicted = predicted[0] + predicted[1] + predicted[2]
    # 推断为真笔迹还是伪笔迹
    predicted[predicted < 2] = 0
    predicted[predicted >= 2] = 1
    predicted = predicted.view(-1)
    # 计算准确率
    accuracy = torch.sum(predicted == labels).item() / labels.size()[0]
    return accuracy

# 设置超参数
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.001
# 设置随机数
np.random.seed(0)
torch.manual_seed(1)

cuda = torch.cuda.is_available()
# 设置训练集和测试集
train_set = dataset(train=True)
test_set = dataset(train=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2*BATCH_SIZE, shuffle=False)
# 构建模型
model = net()
if cuda:
    model = model.cuda()
# 设置损失函数和优化器
criterion = loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
# 设置日志文件
writer = SummaryWriter(log_dir='scalar')
if cuda:
    criterion = criterion.cuda()
iter_n = 0
t = time.strftime("%m-%d-%H-%M", time.localtime())
print(len(train_loader))
# 开始训练
for epoch in range(1, EPOCHS + 1):
    # 输入数据
    for i, (inputs, labels) in enumerate(train_loader):
        torch.cuda.empty_cache()
        print(inputs.shape)

        optimizer.zero_grad()

        labels = labels.float()
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        # 得到模型输出结果
        predicted = model(inputs)
        # 计算损失函数
        loss = criterion(*predicted, labels)  
        
        loss.backward()
        optimizer.step()
        # 计算准确率
        accuracy = compute_accuracy(predicted, labels)

        writer.add_scalar(t+'/train_loss', loss.item(), iter_n)
        writer.add_scalar(t+'/train_accuracy', accuracy, iter_n)
        # 输出记录测试结果及准确率
        if i % 100 == 0 and i :
            with torch.no_grad():
                accuracys = []
                for i_, (inputs_, labels_) in enumerate(test_loader):
                    labels_ = labels_.float()
                    if cuda:
                        inputs_, labels_ = inputs_.cuda(), labels_.cuda()
                    predicted_ = model(inputs_)
                    accuracys.append(compute_accuracy(predicted_, labels_))
                accuracy_ = sum(accuracys) / len(accuracys)
                writer.add_scalar(t+'/test_accuracy', accuracy_, iter_n)
            print('test accuracy:{:.6f}'.format(accuracy_))

        iter_n += 1
        # 保存模型
        if i == 100:
            torch.save(model.state_dict(), 'model.pth')

        if i % 10 == 0:
            print('Epoch[{}/{}], iter {}, loss:{:.6f}, accuracy:{}'.format(epoch, EPOCHS, i, loss.item(), accuracy))

writer.close()
