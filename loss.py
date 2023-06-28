import torch
import torch.nn as nn

class loss(nn.Module): # 设置损失函数
    def __init__(self):
        super(loss, self).__init__()
        # 选择BCEloss作为评价标准
        self.bce_loss = nn.BCELoss()

    
    def forward(self, x, y, z, label):
        # 调整三部分权重
        alpha_1, alpha_2, alpha_3 = 0.3, 0.4, 0.3
        label = label.view(-1, 1)
        # print(max(x), max(label))
        # 得到三组对照组的loss结果
        loss_1 = self.bce_loss(x, label)
        loss_2 = self.bce_loss(y, label)
        loss_3 = self.bce_loss(z, label)
        # 计算平均loss
        return torch.mean(alpha_1*loss_1 + alpha_2*loss_2 + alpha_3*loss_3)