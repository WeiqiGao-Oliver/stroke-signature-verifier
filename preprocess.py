import cv2
import os
import numpy as np
import random

def resize1(path):
	# 重新调整图像大小，调整伪字迹
	try:
		img = cv2.imread('CEDAR/full_forg/' + path, 0)
		dst = cv2.resize(img, (220, 155), cv2.INTER_LINEAR)
		cv2.imwrite('CEDAR/full_forg_gray_115x220/{}'.format(path), dst)
	except:
		print(path)

path = 'CEDAR/full_forg'
for p in os.listdir(path):
	resize1(p)

def resize2(path):
	# 重新调整图像大小，调整真字迹
	try:
		img = cv2.imread('CEDAR/full_org/' + path, 0)
		dst = cv2.resize(img, (220, 155), cv2.INTER_LINEAR)
		cv2.imwrite('CEDAR/full_org_gray_115x220/{}'.format(path), dst)
	except:
		print(path)

path = 'CEDAR/full_org'
for p in os.listdir(path):
	resize2(p)
# 划分训练集和测试集，以10:1的比例调整训练集和测试集，并成对组合，其中需要伪字迹和真字迹形成对照组。
with open('CEDAR/gray_train.txt', 'w') as f:
	for i in range(1, 51):
		for j in range(1, 25):
			for k in range(j+1, 25):
				f.write('full_org_gray_115x220/original_{0}_{1}.png full_org_gray_115x220/original_{0}_{2}.png 1\n'.format(i, j, k))
		org_forg = [(j,k) for j in range(1, 25) for k in range(1, 25)]
		for (j, k) in random.choices(org_forg, k=276):
			f.write('full_org_gray_115x220/original_{0}_{1}.png full_forg_gray_115x220/forgeries_{0}_{2}.png 0\n'.format(i, j, k))

with open('CEDAR/gray_test.txt', 'w') as f:
	for i in range(51, 56):
		for j in range(1, 25):
			for k in range(j+1, 25):
				f.write('full_org_gray_115x220/original_{0}_{1}.png full_org_gray_115x220/original_{0}_{2}.png 1\n'.format(i, j, k))
		org_forg = [(j,k) for j in range(1, 25) for k in range(1, 25)]
		for (j, k) in random.choices(org_forg, k=276):
			f.write('full_org_gray_115x220/original_{0}_{1}.png full_forg_gray_115x220/forgeries_{0}_{2}.png 0\n'.format(i, j, k))

