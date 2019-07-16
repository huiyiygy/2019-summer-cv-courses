# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:DataAugmentation.py
@time:2019/07/05 17:50
"""
import cv2 as cv
import numpy as np
import random


class DataAugmentation:
    @staticmethod
    def crop(img, x, y, crop_width, crop_height):
        """
        裁剪图片
        Inputs
        ------
        - img: 读入的numpy图片
        - x: 裁剪起始点横坐标
        - y: 裁剪起始点纵坐标
        - width: 裁剪图片宽度
        - height: 裁剪图片高度
        Return
        ------
        - result 裁剪后的图片
        """
        height, width = img.shape[0], img.shape[1]
        if x < 0 or crop_width <= 0 or x + crop_width > width or y < 0 or crop_height <= 0 or y + crop_height > height:
            raise Exception("Invalid parameters!")
        result = img[y:y+crop_height, x:x+crop_width, :].copy()
        return result

    @staticmethod
    def color_shift(img, low_bound=-50, high_bound=50):
        """
        随机改变图片颜色
        Inputs
        ------
        - img: 读入的numpy图片
        - low_bound: 处理像素灰度值的边界
        - high_bound:
        Return
        ------
        - result 灰度迁移后的图片
        """
        result = img.copy()
        channels = cv.split(result)
        for channel in channels:
            color = random.randint(low_bound, high_bound)
            if color == 0:
                continue
            elif color > 0:  # Brighten the origin image
                lim = 255 - color
                channel[channel > lim] = 255
                channel[channel <= lim] = (color + channel[channel <= lim]).astype(img.dtype)
            elif color < 0:  # Darken the origin image
                lim = 0 - color
                channel[channel < lim] = 0
                channel[channel >= lim] = (color + channel[channel >= lim]).astype(img.dtype)
        result = cv.merge((channels[0], channels[1], channels[2]))
        return result

    @staticmethod
    def gamma_correction(img, gamma=1.0):
        """
        伽马校正
        Inputs
        ------
        - img: 读入的numpy图片
        - gamma: 伽马系数 默认为 1.0
        Return
        ------
        - result 伽马校正后的图片
        """
        inv_gamma = 1.0 / gamma
        table = []
        for i in range(256):
            table.append(((i / 255.0) ** inv_gamma) * 255)
        table = np.array(table).astype("uint8")
        result = cv.LUT(img, table)
        return result

    @staticmethod
    def histogram_equalized(img, scale=1):
        """
        直方图均衡化
        Inputs
        ------
        - img: 读入的numpy图片
        - scale: 图像缩放比例 默认为 1
        Return
        ------
        - result 直方图均衡化后的图片
        """
        img_resize = cv.resize(img, dsize=(int(scale * img.shape[0]), int(scale * img.shape[1])))
        img_yuv = cv.cvtColor(img_resize, cv.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])  # only for 1 channel
        # convert the YUV image back to RGB format
        result = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)  # y: luminance(明亮度), u&v: 色度饱和度
        return result

    @staticmethod
    def rotation(img, rotation_center, angle=0, scale=1):
        """
        旋转图片
        Inputs
        ------
        - img: 读入的numpy图片
        - rotation_center: 图像旋转的中心
        - angle: 旋转角度 默认为 0
        - scale: 图像缩放比例 默认为 1
        Return
        ------
        - result 旋转后的图片
        """
        rotation_matrix = cv.getRotationMatrix2D(center=rotation_center, angle=angle, scale=scale)
        result = cv.warpAffine(img, rotation_matrix, dsize=(img.shape[1], img.shape[0]))
        return result

    @staticmethod
    def affine_transform(img, src=None, dst=None):
        """
        仿射变换 = 旋转 + 平移
        Inputs
        ------
        - img: 读入的numpy图片
        - src: 三对原图中的点
        - dst: 三对仿射变换后的对应点
        Return
        ------
        - result 仿射变换后的图片
        """
        rows, cols, _ = img.shape
        if src is None:
            src = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        if dst is None:
            dst = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
        # 求仿射变换矩阵
        m = cv.getAffineTransform(src=src, dst=dst)
        result = cv.warpAffine(img, M=m, dsize=(cols, rows))

        return result

    @staticmethod
    def perspective_transform(img, src=None, dst=None):
        """
        透视变换/投影变换：将图像投影到一个新的视平面。仿射变换可以理解为透视变换的特殊形式。
        Inputs
        ------
        - img: 读入的numpy图片
        - src: 四对原图中的点
        - dst: 四对透视变换后的对应点
        Return
        ------
        - result 透视变换后的图片
        """
        height, width, _ = img.shape
        if src is None or dst is None:
            # warp:
            margin = random.randint(0, int(min(height, width)/2))
            x1 = random.randint(-margin, margin)
            y1 = random.randint(-margin, margin)
            x2 = random.randint(width - margin - 1, width - 1)
            y2 = random.randint(-margin, margin)
            x3 = random.randint(width - margin - 1, width - 1)
            y3 = random.randint(height - margin - 1, height - 1)
            x4 = random.randint(-margin, margin)
            y4 = random.randint(height - margin - 1, height - 1)

            dx1 = random.randint(-margin, margin)
            dy1 = random.randint(-margin, margin)
            dx2 = random.randint(width - margin - 1, width - 1)
            dy2 = random.randint(-margin, margin)
            dx3 = random.randint(width - margin - 1, width - 1)
            dy3 = random.randint(height - margin - 1, height - 1)
            dx4 = random.randint(-margin, margin)
            dy4 = random.randint(height - margin - 1, height - 1)

            src = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            dst = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        # 求透视变换矩阵
        m_warp = cv.getPerspectiveTransform(src, dst)
        result = cv.warpPerspective(img, M=m_warp, dsize=(width, height))
        return result
