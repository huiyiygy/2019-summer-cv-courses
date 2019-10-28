# -*- coding: utf-8 -*-
import numpy as np


def iou(A, B):
    # 左上右下坐标(x1,y1,x2,y2)
    w = max(0, min(A[2], B[2]) - max(A[0], B[0]) + 1)
    h = max(0, min(A[3], B[3]) - max(A[1], B[1]) + 1)
    area1 = (A[2] - A[0] + 1) * (A[3] - A[1] + 1)
    area2 = (B[2] - B[0] + 1) * (B[3] - B[1] + 1)
    inter = w * h
    union = area1 + area2 - inter
    return inter / union


def nms(boxes, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = boxes[:, 0]  # xmin
    y1 = boxes[:, 1]  # ymin
    x2 = boxes[:, 2]  # xmax
    y2 = boxes[:, 3]  # ymax
    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # argsort()返回数组值从小到大的索引值
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:  # 还有数据
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        IOU = inter / (areas[i] + areas[order[1:]] - inter)
        # 找到重叠度不高于阈值的矩形框索引
        left_index = np.where(IOU <= thresh)[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[left_index + 1]
    return keep


def soft_nms(dets, sigma=0.5, Nt=0.5, method=2, threshold=0.1):
    box_len = len(dets)   # box的个数
    for i in range(box_len):
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]
        max_pos = i
        max_scores = ts

        # get max box
        pos = i+1
        while pos < box_len:
            if max_scores < dets[pos, 4]:
                max_scores = dets[pos, 4]
                max_pos = pos
            pos += 1

        # add max box as a detection
        dets[i, :] = dets[max_pos, :]

        # swap ith box with position of max box
        dets[max_pos, 0] = tmpx1
        dets[max_pos, 1] = tmpy1
        dets[max_pos, 2] = tmpx2
        dets[max_pos, 3] = tmpy2
        dets[max_pos, 4] = ts

        # 将置信度最高的 box 赋给临时变量
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]

        pos = i+1
        # NMS iterations, note that box_len changes if detection boxes fall below threshold
        while pos < box_len:
            x1, y1, x2, y2 = dets[pos, 0], dets[pos, 1], dets[pos, 2], dets[pos, 3]

            area = (x2 - x1 + 1)*(y2 - y1 + 1)

            iw = (min(tmpx2, x2) - max(tmpx1, x1) + 1)
            ih = (min(tmpy2, y2) - max(tmpy1, y1) + 1)
            if iw > 0 and ih > 0:
                overlaps = iw * ih
                ious = overlaps / ((tmpx2 - tmpx1 + 1) * (tmpy2 - tmpy1 + 1) + area - overlaps)

                if method == 1:    # 线性
                    if ious > Nt:
                        weight = 1 - ious
                    else:
                        weight = 1
                elif method == 2:  # gaussian
                    weight = np.exp(-(ious**2) / sigma)
                else:              # original NMS
                    if ious > Nt:
                        weight = 0
                    else:
                        weight = 1

                # 赋予该box新的置信度
                dets[pos, 4] = weight * dets[pos, 4]

                # 如果box得分低于阈值thresh，则通过与最后一个框交换来丢弃该框
                if dets[pos, 4] < threshold:
                    dets[pos, 0] = dets[box_len-1, 0]
                    dets[pos, 1] = dets[box_len-1, 1]
                    dets[pos, 2] = dets[box_len-1, 2]
                    dets[pos, 3] = dets[box_len-1, 3]
                    dets[pos, 4] = dets[box_len-1, 4]

                    box_len = box_len-1
                    pos = pos-1
            pos += 1

    keep = [i for i in range(box_len)]
    return keep


if __name__ == '__main__':
    dets = [[0, 0, 100, 101, 0.9], [5, 6, 90, 110, 0.7], [17, 19, 80, 120, 0.8], [10, 8, 115, 105, 0.5]]
    dets = np.array(dets)
    result = soft_nms(dets, 0.5)
    print(result)
    print("%.4f" % iou(dets[0], dets[1]))
