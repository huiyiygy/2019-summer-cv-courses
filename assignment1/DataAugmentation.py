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
    def random_crop(img):
        """
        随机裁剪图片
        - img: 读入的numpy图片
        :return:
        """
        height, width = img.shape[0], img.shape[1]

        x = random.randint(0, width // 2)
        y = random.randint(0, height // 2)

        result = img[y:height - y, x:width - x, :].copy()

        return result

    @staticmethod
    def random_color_shift(img):
        """
        随机改变图片颜色
        - img: 读入的numpy图片
        :return:
        """
        result = img.copy()

        channels = cv.split(result)
        for channel in channels:
            color = random.randint(-50, 50)
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
        :param img: 读入的numpy图片
        :param gamma:
        :return:
        """
        inv_gamma = 1.0 / gamma
        table = []
        for i in range(256):
            table.append(((i / 255.0) ** inv_gamma) * 255)
        table = np.array(table).astype("uint8")
        result = cv.LUT(img, table)
        return result

    @staticmethod
    def histogram_equalized(img):
        """
        直方图均衡化
        :param img: 读入的numpy图片
        :return:
        """
        img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])  # only for 1 channel
        # convert the YUV image back to RGB format
        result = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)  # y: luminance(明亮度), u&v: 色度饱和度

        return result

    @staticmethod
    def random_rotation(img):
        """
        随机旋转图片
        - img:  读入的numpy图片
        :return:
        """
        rotation_point = (img.shape[1] / 2, img.shape[0] / 2)
        angle = random.randint(-180, 180)
        # rotation_point, angle, scale
        rotation_matrix = cv.getRotationMatrix2D(center=rotation_point, angle=angle, scale=1)
        result = cv.warpAffine(img, rotation_matrix, dsize=(img.shape[1], img.shape[0]))

        return result

    @staticmethod
    def affine_transform(img):
        """
        仿射变换
        :param img: 读入的numpy图片
        :return:
        """
        rows, cols, _ = img.shape
        pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

        m = cv.getAffineTransform(pts1, pts2)
        result = cv.warpAffine(img, m, (cols, rows))

        return result

    @staticmethod
    def perspective_transform(img):
        """
        透视变换
        :param img: 读入的numpy图片
        :return:
        """
        height, width, channels = img.shape

        # warp:
        random_margin = random.randint(0, int(min(height, width)/2))
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)

        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        m_warp = cv.getPerspectiveTransform(pts1, pts2)
        result = cv.warpPerspective(img, m_warp, (width, height))

        return result
