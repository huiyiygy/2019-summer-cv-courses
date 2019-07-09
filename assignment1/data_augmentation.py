# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:data_augmentation.py
@time:2019/07/06 18:58
"""
import cv2 as cv
import matplotlib.pyplot as plt

from DataAugmentation import DataAugmentation


def main():
    # Load Image and change image channel from BGR to RGB
    img_path = 'logo.jpg'
    img = cv.imread(img_path)
    cv.cvtColor(img, cv.COLOR_BGR2RGB, img)

    # Define Data Augmentation Tools
    data_augment = DataAugmentation()

    # Random Crop Image
    crop_img = data_augment.random_crop(img)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('origin image')
    ax1.imshow(img)
    ax1.axis('off')
    ax2.set_title('croped image')
    ax2.imshow(crop_img)
    ax2.axis('off')

    # Random Color Shift
    color_shift_img = data_augment.random_color_shift(img)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('origin image')
    ax1.imshow(img)
    ax1.axis('off')
    ax2.set_title('color shifted image')
    ax2.imshow(color_shift_img)
    ax2.axis('off')

    # Gamma Correction
    gamma_correction_img = data_augment.gamma_correction(img, gamma=1.5)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('origin image')
    ax1.imshow(img)
    ax1.axis('off')
    ax2.set_title('gamma correction image')
    ax2.imshow(gamma_correction_img)
    ax2.axis('off')

    # histogram
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(img.flatten(), 256, [0, 256], color='r')
    histogram_equalized_img = data_augment.histogram_equalized(img)  # 直方图均衡化图像
    ax2.set_title('origin image')
    ax2.imshow(img)
    ax2.axis('off')
    ax3.set_title('Histogram Equalized image')
    ax3.imshow(histogram_equalized_img)
    ax3.axis('off')
    plt.tight_layout()

    # Random Rotation
    random_rotation_img = data_augment.random_rotation(img)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('origin image')
    ax1.imshow(img)
    ax1.axis('off')
    ax2.set_title('random rotation image')
    ax2.imshow(random_rotation_img)
    ax2.axis('off')

    # Affine Transform
    affine_transform_img = data_augment.affine_transform(img)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('origin image')
    ax1.imshow(img)
    ax1.axis('off')
    ax2.set_title('affine transform image')
    ax2.imshow(affine_transform_img)
    ax2.axis('off')

    # Perspective Transform
    perspective_transform_img = data_augment.perspective_transform(img)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('origin image')
    ax1.imshow(img)
    ax1.axis('off')
    ax2.set_title('perspective transform image')
    ax2.imshow(perspective_transform_img)
    ax2.axis('off')


if __name__ == '__main__':
    main()
