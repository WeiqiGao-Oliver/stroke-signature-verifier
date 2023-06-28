import torch
import torch.nn as nn

import cv2

# 设置卷积流层
class stream(nn.Module):
	def __init__(self):
		super(stream, self).__init__()
        # 设置初始卷积层，共8层卷积，分为四部分。激活函数采用的ReLU，池化层用的最大池化。
		self.stream = nn.Sequential(
			nn.Conv2d(1, 32, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(32, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(64, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(96, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(96, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2)
			)

	def forward(self, reference, inverse):
		# 依次经过四部分卷积流结构
		for i in range(4):
			# 原始图像经过两层卷积后，等待灰度图进行全部卷积，通过attention进行关联，再经过三层卷积。
			reference = self.stream[0 + i * 5](reference)
			reference = self.stream[1 + i * 5](reference)
			inverse = self.stream[0 + i * 5](inverse)
			inverse = self.stream[1 + i * 5](inverse)
			inverse = self.stream[2 + i * 5](inverse)
			inverse = self.stream[3 + i * 5](inverse)
			inverse = self.stream[4 + i * 5](inverse)
			reference = self.attention(inverse, reference)
			reference = self.stream[2 + i * 5](reference)
			reference = self.stream[3 + i * 5](reference)
			reference = self.stream[4 + i * 5](reference)
			

		return reference, inverse


	def attention(self, inverse, discrimnative):

		GAP = nn.AdaptiveAvgPool2d((1, 1))
		sigmoid = nn.Sigmoid()

		up_sample = nn.functional.interpolate(inverse, (discrimnative.size()[2], discrimnative.size()[3]), mode='nearest')
		conv = getattr(self, 'Conv_' + str(up_sample.size()[1]), 'None')
		g = conv(up_sample)
		g = sigmoid(g)
		tmp = g * discrimnative + discrimnative
		f = GAP(tmp)
		f = f.view(f.size()[0], 1, f.size()[1])
		
		fc = getattr(self, 'fc_' + str(f.size(2)), 'None')
		f = fc(f)
		f = sigmoid(f)
		f = f.view(-1, f.size()[2], 1, 1)
		out = tmp * f

		return out


if __name__ == '__main__':
	model = stream()
	x = {}
