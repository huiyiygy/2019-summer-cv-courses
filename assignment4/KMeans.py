# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:KMeans.py
@time:2019/08/03 19:05
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, k=1, color_map=None, central_ids=None):
        # 数据点集
        self.data = data
        # 类别数
        self.k = k
        # 类别颜色
        self.color_map = color_map
        # 当前中心点坐标
        if central_ids is not None:
            self.central_ids = central_ids
        else:
            self.central_ids = {
                i: [np.random.randint(0, 80), np.random.randint(0, 80)] for i in range(k)
            }
        # 当前所有点所属类别
        self.closest_central_ids = None

    def train(self, iterators=10):
        for i in range(iterators):
            print('-----------------')
            print('Iterator: {}'.format(i+1))
            self._calculate_distance()
            self._update_central_point()
            self.show(iterators=(i+1))

            if self.closest_central_ids is not None and self.closest_central_ids.equals(self.data['closest']):
                break
            # 更新所有点所属类别
            self.closest_central_ids = self.data['closest'].copy(deep=True)
        print('Trainning finished')

    def _calculate_distance(self):
        """
        计算数据点集中每个点到当前各个中心点的距离，根据最近距离将其分配给各个中心点，并分配颜色
        """
        # 计算每个点到各个中心点的距离
        for i in self.central_ids.keys():
            # sqrt((x1 - x2)^2 - (y1 - y2)^2)
            self.data['distance_from_{}'.format(i)] = (
                np.sqrt(np.square(self.data['x'] - self.central_ids[i][0]) +
                        np.square(self.data['y'] - self.central_ids[i][1]))
            )
        distance_from_central_ids = ['distance_from_{}'.format(i) for i in self.central_ids.keys()]
        # 根据最近距离将其分配给各个中心点    取多列数据data.loc[:,['A','B']]
        self.data['closest'] = self.data.loc[:, distance_from_central_ids].idxmin(axis=1)
        # map()用于映射匹配 lstrip()截掉字符串左边的空格或指定字符。
        self.data['closest'] = self.data['closest'].map(lambda x: int(x.lstrip('distance_from_')))
        # 根据类别分配颜色
        self.data['color'] = self.data['closest'].map(lambda x: self.color_map[x])

    def _update_central_point(self):
        """
        更新中心点坐标
        """
        for i in self.central_ids.keys():
            self.central_ids[i][0] = np.mean(self.data[self.data['closest'] == i]['x'])
            self.central_ids[i][1] = np.mean(self.data[self.data['closest'] == i]['y'])

    def show(self, iterators=0):
        plt.scatter(self.data['x'], self.data['y'], color=self.data['color'], alpha=0.5, edgecolor='k')
        for i in self.central_ids.keys():
            plt.scatter(*self.central_ids[i], color=self.color_map[i], linewidths=6)
        plt.title('Iterators: {}'.format(iterators))
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()


def main():
    # step 0.0: generate source data
    # data = pd.DataFrame({
    #     'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
    #     'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    # })
    data = pd.DataFrame({
        'x': [random.randint(0, 80) for _ in range(100)],
        'y': [random.randint(0, 80) for _ in range(100)]
    })
    # step 0.1: generate central point
    np.random.seed(2)
    k = 3
    # step 0.2: assign centroid for each source data
    color_map = {0: 'r', 1: 'g', 2: 'b'}
    kmeans = KMeans(data, k, color_map)
    kmeans.train(iterators=10)


if __name__ == '__main__':
    main()
