# Regreesion 工作记录

## 简介
该目录主要实现基于线性回归的协同过滤算法，完成训练、验证和测试的多种功能
## 使用方法
在本项目**根目录**下，运行以下指令查看各个命令行参数用法含义(不要轻易修改默认参数，除非调的更好)
```shell
python -m Regression.main -h
```
### 训练
```shell
python -m Regression.main --mode train
```
下载部分检查点，下面这个链接是我在实验记录一节中部分检查点(10, 50, 100, 150, 200)的压缩包，
如果希望继续训练，请自行下载解压，将pth文件放在根目录下的`CheckPoints/CFLR/`目录下。

[百度网盘](https://pan.baidu.com/s/1A8UPPWlYs3mMQrzBvujvNw) 
提取码:6kj4

### 验证(目前在训练集上)
```shell
python -m Regression.main --mode eval --ckpt CKPT --rmse --ptopk --rtopk --corr
```
TODO 划分训练集和验证集
### 测试(即生成result.txt)
```shell
python -m Rergession.main --mode test --ckpt CKPT 
```


### 调试
在先前的代码的基础上，选择一个数据集读取的用户数的上限UPPER作为命令行参数，运行前文所述的训练和验证指令

学习率比较难调试，那就用手工的learning rate schedular吧

### 比较
```shell
python -m Regression.comp result_1.txt result_2.txt --rmse --ptopk --rtopk --corr
```

## 实验记录
初始参数
```
Namespace(mode='train', ckpt=0, norm=True, upper=-1, numf=6, lamb=0.1, init=0.5, nepoch=50, lr=0.0003, pfreq=500, seed=20, rmse=False, ptopk=False, rtopk
=False, corr=False, sort=True, thresh=80, k=10)
```
| Epoch | LR | Shuffle | Batch | lambda | LOSS | RMSE | PTOPK(%) | RTOPK(%) | CORR |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 10   |0.0003 | F | 1 | 0.1  |0.6585| 41.35| 0.61 | 0.15 |0.0808|
| 20   |0.0003 | F | 1 | 0.1  |0.1616| 37.98| 1.03 | 0.39 |0.1275|
| 30   |0.0003 | F | 1 | 0.1  |0.1510| 36.55| 1.01 | 0.41 |0.1481|
| 40   |0.0003 | F | 1 | 0.1  |0.1445| 35.65| 1.05 | 0.47 |0.1568|
| 50   |0.0003 | F | 1 | 0.1  |0.1408| 34.97| 1.13 | 0.52 |0.1628|
| 55   |0.0005 | F | 1 | 0.1  |0.1910| 34.99| 1.15 | 0.51 |0.1638|
| 60   |0.0003 | F | 1 | 0.1  |0.1400| 35.03| 1.25 | 0.57 |0.1495|
| 70   |0.0003 | F | 1 | 0.1  |0.1352| 34.09| 1.39 | 0.67 |0.1685|
| 80   |0.0003 | F | 1 | 0.1  |0.1333| 33.69| 1.51 | 0.74 |0.1738|
| 90   |0.0004 | F | 1 | 0.1  |0.1312| 33.00| 1.67 | 0.86 |0.1785|
| 100  |0.0004 | F | 1 | 0.1  |0.1294| 32.52| 1.85 | 0.98 |0.1830|
| 110  |0.0005 | F | 1 | 0.1  |0.1284| 31.98| 2.07 | 1.14 |0.1871|
| 120  |0.0005 | T | 1 | 0.1  |0.1302| 31.99| 2.08 | 1.12 |0.1892|
| 130  |0.0005 | T | 1 | 0.1  |0.1284| 32.90| 2.34 | 1.24 |0.1693|
| 140  |0.0005 | T | 1 | 0.1  |0.1287| 32.08| 2.64 | 1.42 |0.1852|
| 150  |0.0005 | T | 1 | 0.1  |0.1421| 31.77| 2.78 | 1.48 |0.1814|
| 160  |0.0005 | T | 1 | 0.1  |0.1349| 31.37| 3.02 | 1.63 |0.1888|
| 170  |0.0005 | T | 1 | 0.1  |0.1267| 31.01| 3.22 | 1.75 |0.1964|
| 180  |0.0005 | T |100|0.0001|4.7643| 28.56| 5.25 | 3.45 |0.1908|
| 190  |0.0005 | T |100|0.0001|4.3304| 27.70| 5.82 | 3.86 |0.2101|
| 200  |0.0005 | T |100|0.0001|4.2322| 27.07| 6.55 | 4.41 |0.2176|

---

**以下是历史内容**

## 已经完成的
### DataLoader
- 读取`train.txt`和`itemAttribute.txt`，
- 保存了userName与userID的映射关系和itemName和itemID的映射；
- 保存所有userID到`[itemID, score]` List的邻接链表（没打分的不存）， 
- 保存了所有出现了的itemID到其attribute的数组（属性不全的暂时存为0 **改！**），
- `load_data()`接口返回三个矩阵，把稀疏矩阵转化为了稠密矩阵，并且一次只返回所有用户在`[pos, pos+B)`个物品上的打分情况
虽然这样利于后续模型处理的编写，但是实践表明由于矩阵过于稀疏，加之内存限制，计算效率会很低，最好可以重写一个`load_data_2`

### train
`train`是可以通用的训练过程

### ContentLinearModel
`ContentLinearModel`希望为U个用户每人学习D+1个参数$W$，D=2是电影属性$X$的维度，然后$\hat{Y}=WX$估计分数
严格的定义如下：

$r(i,j)$表示用户j是否为电影i打分

$y^{(i,j)}$表示用户j给电影i打的分（如果有的话）

$m^{(j)}$表示用户j打分的电影数量

则目标函数(加入正则化项)为

$$
\min_{\theta^{(j)}}\frac{1}{2m^{(j)}}\sum_{i:r(i,j)=1}(\theta^{(j)T}x^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2m^{(j)}}\sum_{k=1}^n(\theta_k^{(j)})^2
$$

## 存在的问题
训练过程loss不断升高，原因能是以下几点：
1. 有些电影的属性不知道但是目前却直接赋值为0，这使得`y_hat`中有大量的0；
2. 属性值很大，应当在D维度方向做归一化，否则`y_hat`非零项远远超过100；
3. 如果改善1、2后仍难以收敛可能需要对分数在做归一化；
4. 数据本身不是线性关系，尝试使用高斯核和多项式核？；

## TODO List
- [ ] 解决问题1、2 (估计半天)
- [x] 写线性回归的协同过滤模型(就不要向量化了，多重for循环吧)，并完成与之对应的`load_data`(估计一天)
- [ ] 解决问题3、4 (未知)
- [ ] 如果基于内容的效果好，可以给协同过滤的模型加两个维度，使用前者训练好的参数作为预训练 (估计没空)

## 参考链接
[吴恩达-机器学习-协同过滤（比较通俗易懂）](https://www.bilibili.com/video/BV164411b7dx?p=98)