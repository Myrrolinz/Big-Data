## 文件介绍

### 文件夹

`Save`文件夹保存训练好的模型、测试好的模型以及保存的相似矩阵

### 文件

`cf_class.py`文件实现了基于项目的协同过滤

`user.py`实现了基于用户的协同过滤

`main.py`为模型训练启动保存脚本

`config.py`存有全局的控制信息

`utils.py`存有一些全局公用的方法

## 模型功能

### 基于项目的模型功能

模型所保存的数据，其中，`self.sim_matrix`只保存分块中的一份。

```python
     def __init__(self, train_p, test_p):
        self.if_build = False
        self.if_train = False
        self.if_test = False
        self.rating_num = 0
        self.user_matrix = []  # 存储用户对物品的评分 [{itemid: score,...},...]
        # self.item_matrix = []  # 存储物品的属性 [[at1,at2],...]  now useless
        self.user_ave = []  # 用户对物品的评分准则(对物品评分的平均数)[u1,u2,...]
        # self.user_item_index = []  # 索引矩阵[[item1,item2,...],[],...]
        self.item_user_index = []  # 反向索引[{user: score, ...},...]
        self.sim_matrix = None  # item 的 相似矩阵（稀疏）lil_matrix
        self.item_list = set()
        self.change = dict()
        self.r = []  # predicted matirx
        # self.sim_csr = [[], [], []]  # item 的相似矩阵 使用csr的方式进行压缩 [[row_offset,...], [col,...],[value,...]]
        self.train_p = train_p
        self.test_p = test_p
        self.total = 0
        self.now_sim = 0
```

模型建立与训练

```python
def build(self, path):
"""
使用初始化时输入的文件路径，建立基本信息
"""

def train(self):
"""
使用build中获得的信息进行训练，建立self.sim_matrix，实现了矩阵分块
"""
```

测试

```python
def test(self, path):
"""
使用测试文件与相似矩阵进行测试
"""
def get_sim(self, i, j):
"""
获得相似矩阵元素，i，j分别为行、列
"""
    return self.sim_matrix[self.get_offset(i, j)]
def predict(self, user, item_j):

"""
根据用户id与物品id进行预测
"""
def get_offset(self, i, j):

"""
根据行列再分块矩阵中检索
"""

    ofset = int(i*(len(self.item_list)) + j - (i+1)*(i+2)/2)
    if ofset / pow(10, save_per) == self.now_sim:
        return ofset - self.now_sim * pow(10, save_per)
    else:
        self.now_sim = ofset / pow(10, save_per)
        self.sim_matrix = load_class(Save_path, os.path.join(Save_path, 'sim'+str(self.now_sim)+'.pickle'))
        return ofset - self.now_sim * pow(10, save_per)
```

### 基于用户的模型

基于用户的模型所使用的模块与基于项目的模型相同，不过再使用公式上进行修改

```python
 def __init__(self, train_p, test_p):
        self.if_build = False
        self.if_train = False
        self.if_test = False
        self.rating_num = 0
        self.user_matrix = []  # 存储用户对物品的评分 [{itemid: score,...},...]
        # self.item_matrix = []  # 存储物品的属性 [[at1,at2],...]  now useless
        self.user_ave = []  # 用户对物品的评分准则(对物品评分的平均数)[u1,u2,...]
        # self.user_item_index = []  # 索引矩阵[[item1,item2,...],[],...]
        self.item_user_index = []  # 反向索引[{user: score, ...},...]
        self.sim_matrix_user = None  # user 的 相似矩阵（稀疏）lil_matrix
        self.item_list = set()
        self.change = dict()
        self.r = []  # predicted matirx
        # self.sim_csr = [[], [], []]  # item 的相似矩阵 使用csr的方式进行压缩 [[row_offset,...], [col,...],[value,...]]
        self.train_p = train_p
        self.test_p = test_p
        self.total = 0
        self.sim1 = 'sim1.user'
        self.sim2 = 'sim2.user'
        self.mid = 0
        self.if_2 = False
        self.now_sim = 0
```
