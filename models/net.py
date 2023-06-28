import torch
import torch.nn as nn
from models.stream import stream
import torchvision

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # 模型结构由stream结构、GAP以及分类层组成。
        self.stream = stream()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        

    def forward(self, inputs):
        half = inputs.size()[1] // 2
        reference = inputs[:, :half, :, :] # 因为我们的输入是级联的结果，所以通过half取前面的一半为refer_img
        reference_inverse = 255 - reference # 灰度图为反向的结果
        test = inputs[:, half:, :, :] # 取另一半为test_img的结果
        del inputs
        test_inverse = 255 - test # 取test_img的灰度图
        # 分别输入到卷积流层当中，得到卷积后的结果
        reference, reference_inverse = self.stream(reference, reference_inverse) 
        test, test_inverse = self.stream(test, test_inverse)
        # 根据三组对照组输出结果
        cat_1 = torch.cat((test, reference_inverse), dim=1)
        cat_2 = torch.cat((reference, test), dim=1)
        cat_3 = torch.cat((reference, test_inverse), dim=1)

        del reference, reference_inverse, test, test_inverse
        # subforward为三个对照组经过GAP层和分类层的输出
        cat_1 = self.sub_forward(cat_1)
        cat_2 = self.sub_forward(cat_2)
        cat_3 = self.sub_forward(cat_3)

        return cat_1, cat_2, cat_3
    
    def sub_forward(self, inputs):
        out = self.GAP(inputs)
        out = out.view(-1, inputs.size()[1])
        out = self.classifier(out)

        return out

if __name__ == '__main__':
    # 模型测试函数
    net = net()
    x = torch.ones(1, 3, 32, 32)
    y = torch.ones(1, 3, 32, 32)
    x_ = torch.ones(1, 3, 32, 32)
    y_ = torch.ones(1, 3, 32, 32)
    out_1, out_2, out_3 = net(x, y, x_, y_)
    # vgg = torchvision.models.vgg13()
    # print(vgg)
