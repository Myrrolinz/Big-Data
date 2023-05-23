# 文件介绍

## 文件夹

`ckpt_files`为本次实验存放checkpoint的文件夹，其中命名规范为：Epoch_P.ckpt和Epoch_Q.ckpt

## 文件

`config.py`为存放过程中使用到的文件夹、文件名等
`LFM.py`为本次实验实现的基于SVD的隐语义模型主体
`main.py`为本次实验的入口程序

# LFM功能函数介绍

构造LFM:

```python
def __init__(self, factor=5, iter_num=10, alpha=0.002, Lambda=0.04, epsilon=1e-2, test_flag=False, keep_last=2)
'''
factor: 因子个数
iter_num: 本次训练的迭代轮数
alpha、Lambda: 超参数
epsilon: 损失值判定条件，当小于此值时提前结束迭代
test_flag: 开发时测试使用，为True时使用小的切分数据集测试；为False时使用正式数据集进行训练
keel_last: 要保留的最新checkpoint数量
'''
```

加载数据集：

```python
def load_data(self)
```

初始化分解矩阵：

```python
def init_latent_matrix(self, init=False)
'''
init: 为False时表示读取最新的checkpoint；为True时表示随机生成分解矩阵，从头开始训练
'''
```

计算test.txt中的结果：

```python
def get_test_result(self)
```

保存checkpoint：

```python
def save_result(self, file_name, data)
'''
file_name: 要保存的文件名
data: 要保存的数据
'''
```

训练：

```python
def train(self)
```

显示预测结果：

```python
def get_pred_result(self, i=-1, j=-1)
'''
x和y均为-1时表示返回整个矩阵的预测结果
x和y均不为-1且合乎规范时，返回用户i对物品j的预测评分
注：返回的评分范围为0-100之间，且评分均为整十（如，10，20，30）
'''
```