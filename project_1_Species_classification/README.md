# 项目一
## 任务一：动物“纲”分类
分别以backbone为Xception和Inceptionv4的预训练网络进行分类训练。

训练参数如下：
- learn-rate 0.001 
- weight-decay 0.0 
- epochs 50 
- batch-size 31 
- input-size 299
- loss 交叉熵损失

训练结果如下：

| Epoch                  | 50     |
| ---------------------- | ------ |
| Xception-pretrained    | 0.95   |
| Inceptionv4-pretrained | 0.83   |

## 任务二：动物“种”分类
分别以backbone为Xception和Inceptionv4的预训练网络进行分类训练。

训练参数如下：
- learn-rate 0.001 
- weight-decay 0.0 
- epochs 50 
- batch-size 31 
- input-size 299
- loss 交叉熵损失

训练结果如下：

| Epoch                  | 50       | 
| ---------------------- | -------- |
| Xception-pretrained    | 0.8625   |
| Inceptionv4-pretrained | 0.7000   | 

## 任务三：动物“纲”,“种”,多任务分类训练
以backbone为Xception的预训练网络进行多任务分类训练。

训练参数如下：
- learn-rate 0.001 
- weight-decay 0.0 
- epochs 50 
- batch-size 31 
- input-size 299
- loss 交叉熵损失

训练结果如下：

| Epoch                  | 100                                                | 
| ---------------------- | -------------------------------------------------- |
| Xception-pretrained    | Acc:0.8875 Acc_classes:0.9375 Acc_species:0.8875   |
