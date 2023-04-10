# PageRank算法实现

## 目录结构

```
├─Block-Stripe             # Block-Stripe方法
    ├─Data                 # Block-Stripe方法的过程文件
    ├─block_stripe.py      # Block-Stripe的python脚本
    └─block_stripe.exe     # Block-Stripe的执行程序
├─Data              # 数据集目录
    └─data.txt      # 本次实验使用的数据集，存储格式为utf-16
├─ReFactor          # 基本Pagerank方法和严格的分块Pagerank方法
    ├─Middle        # 基本Pagerank方法和严格的分块Pagerank方法的过程文件
    ├─basic.py      # 基本Pagerank的python脚本
    ├─basic.exe     # 基本Pagerank的执行程序
    ├─stripes.py    # 严格的分块Pagerank的python脚本
    └─stripes.exe   # 严格的分块Pagerank的执行程序
└─Results                           # 最终结果
    ├─basic.txt                     # 基本Pagerank生成的最终结果
    ├─stripes.txt                   # 严格的分块Pagerank生成的最终结果
    └─Block_stripe_result.txt       # Block-Stripe生成的最终结果
```

## 使用方法
1. 基本Pagerank方法：在`ReFactor`文件夹下运行`basic.exe`
2. 严格的分块Pagerank方法：在`ReFactor`文件夹下运行`stripes.exe`
3. Block-Stripe方法：在`Block-Stripe`文件夹下运行`block_stripe.exe`