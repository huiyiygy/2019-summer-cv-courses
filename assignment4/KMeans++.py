# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:KMeans++.py
@time:2019/08/03 21:15
"""
import numpy as np
import pandas as pd
from assignment4.KMeans import KMeans


class KMeansPlus(KMeans):
    def __init__(self, data: pd.DataFrame, k: int = 1, color_map: dict = None, low=0, high=80):
        central_ids = self._init_central_ids(data, k)
        super(KMeansPlus, self).__init__(data=data, k=k, color_map=color_map, central_ids=central_ids, low=low, high=high)

    @staticmethod
    def _init_central_ids(data: pd.DataFrame, k: int) -> dict:
        # step1 随机选取一个初始中心点
        size = len(data['x'])
        index = np.random.randint(size)
        central_ids = {0: [data['x'][index], data['y'][index]]}
        for i in range(1, k):
            # step2 计算每个样本与当前已有聚类中心之间的最短距离distance
            distance_all = np.full(shape=(size, i), fill_value=np.inf)
            for j in range(i):
                distance_all[:, j] = np.sqrt(np.square(data['x']-central_ids[j][0]) +
                                             np.square(data['y']-central_ids[j][1])
                                             )
            distance = np.min(distance_all, axis=1, keepdims=True)
            distance_square = np.square(distance)
            distance_square_sum = np.sum(distance_square, keepdims=True)
            # 每个样本被选为下一个聚类中心的概率
            p = distance_square / distance_square_sum
            # step3 轮盘法选择下一个聚类中心，步骤：随机产生一个0~1之间的随机数，
            # 判断其在概率累加和中所属的区间，该区间对应的序号就是被选择出来的下一个聚类中心
            p_sum = np.cumsum(p)  # 概率p的累加和
            choice = np.random.rand()
            p_sum = p_sum - choice
            for j in range(size):
                if p_sum[j] <= 0:
                    continue
                else:
                    central_ids[i] = [data['x'][j], data['y'][j]]
                    break
        return central_ids


def main():
    # data = pd.DataFrame({
    #     'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
    #     'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 68, 19, 67, 24, 77]
    # })
    low = -100
    high = 100
    data = pd.DataFrame({
        'x': [np.random.randint(low, high) for _ in range(100)],
        'y': [np.random.randint(low, high) for _ in range(100)]
    })
    k = 3
    color_map = {0: 'r', 1: 'g', 2: 'b'}

    kmeans_plus = KMeansPlus(data, k, color_map, low, high)
    kmeans_plus.train(iterators=10)


if __name__ == '__main__':
    main()
