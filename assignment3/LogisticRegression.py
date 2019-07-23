# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:LogisticRegression.py
@time:2019/07/21 21:35
"""
import numpy as np


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def sigmod(x: np.array):
    return 1.0 / (1.0 + np.exp(-x))


class LogisticRegression:
    """
    A LogisticRegression network with the following architecture:

    x - affine - sigmoid - y

    The network operates on minibatches of data that have shape (n, d) consisting of n samples, each with d features
    """
    def __init__(self, input_dim=100, weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network
        Inputs:
        -------
        - input_dim: An integer giving the size of the input
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random initialization of the weights.
        """
        self.reg = reg
        self.params = {'w': weight_scale * np.random.randn(input_dim, 1), 'b': np.zeros(1)}

    def loss(self, x: np.array, y: np.array = None):
        """
        Evaluate loss and gradient for the LogisticRegression network.
        Inputs:
        -------
        - X: Array of input data of shape (n, d)
        - y: Array of labels, of shape (n,).
        Returns:
        --------
        If y is None, then run a test-time forward pass of the model and return:
        - probability: Array of shape(N,) givging classification probability

        If y is not None, then run a training-time forward and backward pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameters names to
            gradients of the loss with respect to those parameters.
        """
        w, b = self.params['w'], self.params['b']
        n, _ = x.shape

        hx = x.dot(w) + b
        p = sigmod(hx)

        # If y is None then we are in test mode so just return probability
        if y is None:
            return p

        loss, grads = 0.0, {}
        # 将y从(n,) resize为(n, 1)，否则后续与p(n,) 进行对应项相乘时结果为(n, n)，直接导致后续Loss和梯度计算错误
        y = np.resize(y, (len(y), 1))

        a1 = -np.log(p)*y
        a2 = np.log(1-p)*(1-y)
        loss = np.sum(a1 - a2)
        loss = loss / n + 0.5 * self.reg * np.sum(np.square(w))

        # 计算d_p
        d_p = -y / p + (1-y)/(1-p)
        d_p /= n
        # 计算 d_hx
        d_hx = d_p * p * (1 - p)
        # d_hx 快捷计算方式
        # d_hx = (p - y) / n
        # 计算dw db
        dw = x.T.dot(d_hx)
        db = np.sum(d_hx, axis=0)
        # 权值矩阵加上正则化惩罚
        grads['w'] = dw + self.reg * w
        grads['b'] = db
        return loss, grads


def train(x_list: np.array, gt_y_list: np.array, batch_size: int, learn_rate: float, max_iter: int):
    """
    逻辑回归拟合
    Inputs
    ------
    - x_list: shape(n, d) dtype=np.float32
    - gt_y_list: shape(n,) ground truth y list dtype=np.float32
    - num_classes: 类别数
    - batch_size: 批大小 int
    - learn_rate: 学习率 float
    - max_iter: 最大迭代次数 int
    """
    n, d = x_list.shape
    model = LogisticRegression(input_dim=d)
    for i in range(max_iter):
        # 随机抽取数据
        batch_index = np.random.choice(n, batch_size)
        batch_x = x_list[batch_index]
        batch_y = gt_y_list[batch_index]
        # 计算loss和权值梯度
        loss, grads = model.loss(batch_x, batch_y)
        print('iter: %03d, loss: %.4f' % (i, np.mean(loss)))
        # 更新权值
        model.params['w'] -= learn_rate * grads['w']
        model.params['b'] -= learn_rate * grads['b']


def run():
    """
    逻辑回归拟合演示
    """
    n = 100  # 样本数
    d = 50
    c = 2  # 类别数
    # x = np.array([[np.random.uniform(9), np.random.uniform(9)] for _ in range(n)])
    # y = np.zeros((n,))
    # sum_x = np.sum(x, axis=1)
    # for i in range(n):
    #     if sum_x[i] > 10:
    #         y[i] = 1
    x = np.random.randn(n, d)
    y = np.random.randint(c, size=(n,))

    batch_size = 50
    lr = 0.3
    max_iter = 1000
    train(x, y, batch_size, lr, max_iter)


def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
    """
    a native implementation of numerical gradient of f at x
    使用梯度的定义，计算在x点的数值梯度
    f(x)' =  (f(x-h)' - f(x+h)') / 2h
    Inputs:
    -------
    - f: should be a function that takes a single argument
    - x: is the point(numpy array) to evaluate the gradient at
    - verbose: whether  print each gradient in x
    - h: 一个极小的增量
    Returns:
    --------
    - grad: the numerical gradient of f at x
    """
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment
        fxph = f(x)  # evaluate f(x+h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x-h)
        x[ix] = oldval  # restore
        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension
    return grad


def check_loss_and_gradient():
    """
    梯度检查
    :return:
    """
    np.random.seed(1)
    N, D, C = 3, 50, 2
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0.0, 3.14]:
        print('Running check with reg = ', reg)
        model = LogisticRegression(input_dim=D, reg=reg)

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        # Most of the errors should be on the order of e-9 or smaller.
        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))


if __name__ == "__main__":
    # check_loss_and_gradient()
    run()
