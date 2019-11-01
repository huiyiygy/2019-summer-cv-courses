# 项目一
## 任务一：动物“纲”分类
分别以backbone为Xception和Inceptionv4的预训练网络进行分类训练。

训练参数如下：
- weight-decay 0.0 
- epochs 50 
- batch-size 31 
- input-size 299
- loss 交叉熵损失

训练结果如下：

| 模型                   | learnning rate | Accuracy   |
| ---------------------- | -------------- | ---------- |
| Xception-pretrained    | 1e-3           | 0.95       |
| Xception-pretrained    | 1e-4           | **0.9833** |
| Inceptionv4-pretrained | 1e-3           | 0.8333     |
| Inceptionv4-pretrained | 1e-4           | **0.9833** |

## 任务二：动物“种”分类
分别以backbone为Xception和Inceptionv4的预训练网络进行分类训练。

训练参数如下：
- weight-decay 0.0 
- epochs 50 
- batch-size 31 
- input-size 299
- loss 交叉熵损失

训练结果如下：

| 模型                   | learnning rate | Accuracy   |
| ---------------------- | -------------- | ---------- |
| Xception-pretrained    | 1e-3           | 0.8625     |
| Xception-pretrained    | 1e-4           | 0.9250     |
| Inceptionv4-pretrained | 1e-3           | 0.7000     |
| Inceptionv4-pretrained | 1e-4           | **0.9375** |

## 任务三：动物“纲”,“种”,多任务分类训练
以backbone为Xception的预训练网络进行多任务分类训练。

训练参数如下：
- batch-size 31 
- input-size 299
- loss 交叉熵损失

训练结果如下：

| 模型              | Epoch                 | learnning rate | weight-decay          | Accuracy                        | Classes  Accuracy      | Species Accuary         |
| ---------------------------- | -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- |
| Xception-pretrained | 100 | 1e-3 | 0.0  | 0.8875 | 0.9375 | 0.8875 |
| Xception-pretrained | 200 | 1e-3 | 1e-5 | 0.8750 | 0.925 | 0.8750 |
| Xception-pretrained | 200 | 1e-4 | 0.0 | **0.925** | **0.9375** | **0.925** |

