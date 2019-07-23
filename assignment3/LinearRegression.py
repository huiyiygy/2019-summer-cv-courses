# -*- coding:utf-8 -*-
"""
@function: Linear Regression
@author:HuiYi or 会意
@file:LinearRegression.py
@time:2019/07/21 21:34
"""
import numpy as np


def eval_loss(w: float, b: float, x_list: np.array, gt_y_list: np.array):
    """
    loss function
    Inputs
    ------
    - w: 权重 float
    - b: 偏置 float
    - x_list: 输入 shape(n,) dtype=np.float32
    - gt_y_list: 真实值 shape(n,) dtype=np.float32
    Returns
    -------
    - loss: shape(n,) dtype=np.float32
    """
    loss = 0.5 * np.sum((w * x_list + b - gt_y_list) ** 2)
    loss /= len(x_list)
    return loss


def inference(w: float, b: float, x: np.array):
    """
    inference, test, predict, same thing. Run model after training
    Inputs
    ------
    - w: 权重 float
    - b: 偏置 float
    - x: 输入 shape(n,) dtype=np.float32
    Returns
    -------
    - pred_y: 预测值 shape(n,) dtype=np.float32
    """
    pred_y = w * x + b
    return pred_y


def gradient(pred_y: np.array, gt_y: np.array, x: np.array):
    """
    Inputs
    ------
    - pred_y: 预测值 shape(n,) dtype=np.float32
    - gt_y: 真实值 shape(n,) dtype=np.float32
    - x: 输入 shape(n,) dtype=np.float32
    Returns
    -------
    - dw: w的梯度 float
    - db: b的梯度 float
    """
    diff = np.sum(pred_y - gt_y) / pred_y.shape[0]
    dw = diff * x
    db = diff
    return dw, db


def cal_step_gradient(batch_x_list: np.array, batch_gt_y_list: np.array, w: float, b: float, learn_rate: float) -> list:
    """
    单步迭代
    Inputs
    ------
    - batch_x_list: 单步迭代数据 shape(batch_size,) dtype=np.float32
    - batch_gt_y_list: shape(batch_size,) dtype=np.float32
    - w: 权重 float
    - b: 偏置 float
    - learn_rate: 学习率 float
    Returns
    -------
    - w: 单步更新后的权重 float
    - b: 单步更新后的偏置 float
    """
    pred_y = inference(w, b, batch_x_list)
    dw, db = gradient(pred_y, batch_gt_y_list, batch_x_list)
    w -= learn_rate * dw
    b -= learn_rate * db
    return [w, b]


def train(x_list: np.array, gt_y_list: np.array, batch_size: int, learn_rate: float, max_iter: int):
    """
    线性回归拟合直线
    Inputs
    ------
    - x_list: shape(n,) dtype=np.float32
    - gt_y_list: shape(n,) ground truth y list dtype=np.float32
    - batch_size: 批大小 int
    - learn_rate: 学习率 float
    - max_iter: 最大迭代次数 int
    """
    w = 1.0
    b = 0.0
    for i in range(max_iter):
        batch_index = np.random.choice(len(x_list), batch_size)
        batch_x = x_list[batch_index]
        batch_y = gt_y_list[batch_index]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, learn_rate)
        loss = eval_loss(w, b, batch_x, batch_y)
        # print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(np.mean(loss)))


def gen_sample_data() -> list:
    """
    随机生成训练数据
    Returns
    -------
    a list of:
    - x_list: shape(n,) dtype=np.float32
    - y_list: shape(n,) dtype=np.float32
    - w: 权重 float
    - b: 偏置 float
    """
    w = np.random.randint(0, 10) + np.random.randn()
    b = np.random.randint(0, 5) + np.random.randn()
    num_samples = 100
    x_list = np.random.rand(num_samples, 1) * 100
    y_list = w * x_list + b + np.random.randn()
    return [x_list, y_list, w, b]


def run():
    """
    线性回归拟合演示
    """
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.0001
    max_iter = 10000
    train(x_list, y_list, 50, lr, max_iter)


if __name__ == "__main__":
    run()
