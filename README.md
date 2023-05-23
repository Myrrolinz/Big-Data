# 期末大作业说明

## 文件存放结构

```
┝CF_item  # 基于用户和基于物品的协同过滤算法实现目录
├Data  # 存放数据的文件夹
    ├train.txt  # 本次实验使用到的训练集
    ├test.txt   # 本次实验验证效果的测试集
    ├my_train.txt  # 切分出的非常小的训练集，用来测试
    ├my_test.txt   # 切分出的非常小的测试集，用来测试
    └itemAttribute.txt  # Attribute
┝Evaluation  # 用于评估实验效果的脚本集
┝LFM  # 基于SVD的隐语义模型实现目录
┝Regression  # 基于多元线性回归的协同过滤模型实现目录
┝Results  # 保存最终的结果
└Test  # 测试评估脚本的脚本文件夹
```

具体各部分目录下详细内容见各目录中的`README.md`文件

## [Evaluation模块使用方法](Evaluation/README.md)

## 实现方法

- [基于用户与基于物品的协同过滤](CF_item/README.md)
- [基于多元线性回归的协同过滤](Regression/README.md)
- [基于SVD的隐语义模型实现](LFM/README.md)

## 数据集格式

### train.txt

```
train.txt
<user id>|<numbers of rating items>
<item id>   <score>
```

### test.txt

```
test.txt
<user id>|<numbers of rating items>
<item id>
```

### itemAttribute.txt

```
item.txt
<item id>|<attribute_1>|<attribute_2>('None' means this item is not belong to any of attribute_1/2)
```

### result

```
<user id>|<numbers of rating items>
<item id>   <score>

example:
0|6
550452  90  
323933  100  
159248  100  
554099  100  
70896  100  
518385  100 
```

### 可执行文件链接

`链接：https://pan.baidu.com/s/1oDvRzqdMz2xSwJqxX6AUhA?pwd=wejh 
提取码：wejh 
--来自百度网盘超级会员V3的分享`