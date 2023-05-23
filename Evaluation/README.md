## 注意
- 目前没有明确的规定，计算误差时，使用一个用户的关联商品的真实向量与估计向量进行打分，还是使用一个商品的关联用户的真实向量与估计向量进行打分，考虑到实际使用，我倾向于前者，请根据需要自行调整输入维度顺序
- `EvalUtils`中包含了各种基于`numpy`实现的评估方式，大多数接口接收两种维度的输入对，B*N或者N，B即batch size，是向量化的实现，输出是B或者一个float，**没有在batch维度上做平均**，如有需求请自行实现。具体用法请见注释，或者RTSC，或者联系@ashun989
- 考虑到屏蔽细节，在模块中提供了一个高级封装`evalute`，也可以作为`EvalUtils`的使用示例，可以仿照它实现自己的高级封装
- 在`EvalUtils`中的接口均假设送入的Predicted List和True List的每一项都会被记入误差统计，然而在实际过程中可能在你的模型输出和参考输出之外，还有一个属于**参考输出**的mask，表明“这些用户不是打了0分而是没有打分”，将这些想纳入评估将会造成巨大的误差，因此需要设计mask方法，在开发过程中我大概考虑了3种方式：
 
   1. 为EvalUtils的每个接口添加`mask`参数，并且在向量化的计算中修改——效率高——实现难——Debug难（有一个`numpy.ma`的模块将来可以了解一下）
   2. 在评估之前，使用`EvalUtils.gather`，将`mask`标记的部分聚集到一个新的连续序列中，但是因为每个用户打分的电影数量不同，之后不能在向量化——效率低——实现简单——Debug难（丢失了gather前后itemID之间的映射关系），目前在`evaluate`中就是这种方式
   3. 自行设计如稀疏矩阵等数据结构，这应当与`EvalUtils`的实现解耦

- 可能还有bug，甚至是很难发现的计算错误

## 已经实现的评估方式
- 均方根误差（RMSE）：使用`EvalUtils.RMSE`，参考课程PPT“推荐系统1”
- Precision@K, Recall@K: 使用`EvalUtils.precision`, `EvalUtils.recall`，输入需要从score value转化为index，可以使用`EvalUtils.value2id`，[参考](https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54)，需要使用一个阈值过滤出大于某个分数的项目（应用中会被推荐给用户的），在此基础上，因为有TopK的限制，我认为还要按照score value的得分从高到低排序对应的index
- Rank Correlation: 目前实现了[Spearman版本](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)的，这是一种基于排名的pearson相关系数








## TODO List
- [x] 实现`EvalUtils.gather`并改写`evalute`