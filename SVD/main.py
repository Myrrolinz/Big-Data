from SVD import *
import psutil

mySVD=SVDModel()
#读取训练集
mySVD.loadTrainSet()
m1=getProcessMemory()
#进行训练
mySVD.train()
m2=getProcessMemory()

#训练完成后——在验证集上测试
begin=time.time()
mySVD.evaluate()
end=time.time()
duration = end - begin
print("评估模型花费时间为：", "%.6f" % duration, "秒")
m3=getProcessMemory()

#在测试集上测试，输出最终result
begin=time.time()
mySVD.predictOnTestDataset()
end=time.time()
duration = end - begin
print("预测花费时间为：", "%.6f" % duration, "秒")
m4=getProcessMemory()

# 获取程序使用的最大内存空间大小(以MB为单位)
memory_usage =max(m1,m2,m3,m4)

print(f"Model training Memory usage(it indicates the maximum memory that a process used): {memory_usage} MB")
