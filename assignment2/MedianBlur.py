# -*- coding:utf-8 -*-
"""
@function: 中值滤波函数
@author:HuiYi or 会意
@file:MedianBlur.py
@time:2019/07/17 15:46
"""
import numpy as np


def median_blur(img: np.array, kernel: tuple = (3, 3), padding_way: str = 'ZERO'):
    """
    中值滤波函数
    Inputs
    ------
    - img: 读入的numpy图片 shape（W, H, C）, C in [1, 3, 4]
    - kernel: 滤波器大小 shape (M, N) M, N为大于1的奇数
    - padding_way: 填充方式 'REPLICA' or 'ZERO'
    Return
    ------
    - result 中值滤波后的图片
    """
    W, H, C = img.shape
    M, N = kernel[0], kernel[1]

    # Padding image
    pad_width = (M - 1) // 2
    pad_height = (N - 1) // 2
    # padding 为每个维度两边填充的宽度，对于多通道图需要在第三个维度上加(0,0)
    if C == 1:
        padding = ((pad_width, pad_width,), (pad_height, pad_height))
    else:
        padding = ((pad_width, pad_width,), (pad_height, pad_height), (0, 0))

    if padding_way == "ZERO":
        img_pad = np.pad(img, padding, mode='constant', constant_values=0)
    elif padding_way == "REPLICA":
        img_pad = np.pad(img, padding, mode='edge')
    else:
        raise Exception("Unsupported padding_way: %s" % padding_way)

    # Find median
    result = np.zeros_like(img, dtype=img.dtype)
    for i in range(W):
        for j in range(H):
            result[i, j] = np.median(img_pad[i:i+M, j:j+N], axis=(0, 1))

    return result


def salt_pepper_noise(image, prob=0.1):
    """
    添加椒盐噪声
    Inputs
    ------
    - img: 读入的numpy图片 shape（W, H, C）, C in [1, 3, 4]
    - prob:噪声比例
    Return
    ------
    - result 加入椒盐噪声后的图片
    """
    result = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.uniform()
            if rdn < prob:
                result[i][j] = 0
            elif rdn > thres:
                result[i][j] = 255
            else:
                result[i][j] = image[i][j]
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    test_img = cv2.imread('logo.jpg')
    cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB, test_img)

    kernel_size = (5, 5)
    padding_ways = 'REPLICA'  # or 'ZERO'

    noise_img = salt_pepper_noise(test_img)

    median_blur_img = median_blur(noise_img, kernel_size, padding_ways)

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title('origin image')
    ax1.imshow(test_img)
    ax1.axis('off')
    ax2.set_title('Noise image')
    ax2.imshow(noise_img)
    ax2.axis('off')
    ax3.set_title('Median Blur image')
    ax3.imshow(median_blur_img)
    ax3.axis('off')
