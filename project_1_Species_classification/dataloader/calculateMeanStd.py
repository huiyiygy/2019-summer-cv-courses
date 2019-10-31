# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np

filepath = '/home/lab/ygy/cv_course_project/Dataset/train/temp'  # 数据集目录
pathDir = os.listdir(filepath)

R_channel = 0
G_channel = 0
B_channel = 0
num = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = Image.open(os.path.join(filepath, filename))
    w, h = img.size
    img_np = np.array(img, dtype=np.float32) / 255
    print('{}, shpae={}'.format(filename, img_np.shape))
    R_channel += np.sum(img_np[:, :, 0])
    G_channel += np.sum(img_np[:, :, 1])
    B_channel += np.sum(img_np[:, :, 2])
    num += w*h

R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num


R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = Image.open(os.path.join(filepath, filename))
    img_np = np.array(img, dtype=np.float32) / 255
    R_channel = R_channel + np.sum((img_np[:, :, 0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img_np[:, :, 1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img_np[:, :, 2] - B_mean) ** 2)

R_var = R_channel / num
G_var = G_channel / num
B_var = B_channel / num

R_std = np.sqrt(R_var)
G_std = np.sqrt(G_var)
B_std = np.sqrt(B_var)

# mean = (0.656963, 0.621670, 0.550278)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
# var = (0.090119, 0.091931, 0.112209)
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
# std = (0.300198, 0.303201, 0.334976)
print("R_std is %f, G_std is %f, B_std is %f" % (R_std, G_std, B_std))
