import numpy as np
# 打开文件，读取数据
with open('./Results/basic.txt', 'r') as f:
    lines = f.readlines()

# 将每一行的两个元素提取出来，放入字典中
result1 = {}
for line in lines:
    a, b = line.strip().split()
    result1[a] = b
    # 对字典按照键排序
sorted_dict1 = {float(k): v for k, v in sorted(result1.items(), key=lambda item: float(item[0]))}
print(sorted_dict1)

# 得到排序后的数组v1（其中只包含pagerank值）
v1 = [sorted_dict1[k] for k in sorted(sorted_dict1.keys())]
v1 = np.array([float(s) for s in v1])

with open('./Results/standard.txt', 'r') as f:
    lines = f.readlines()
result2 = {}
for line in lines:
    a, b = line.strip().split()
    result2[a] = b
sorted_dict2 = {float(k): v for k, v in sorted(result2.items(), key=lambda item: float(item[0]))}
print(sorted_dict2)

v2 = [sorted_dict2[k] for k in sorted(sorted_dict2.keys())]
v2 = np.array([float(s) for s in v2])

# 计算两次结果中pagerank值的L1范数，并取平均值
dist = np.linalg.norm(v1 - v2, 1) / 100

print(dist)