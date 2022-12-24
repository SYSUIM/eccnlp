环境：tensorflow 2.5

# lambdarank.py
参考https://github.com/Jabberwockleo

# merge_reasons.py
将uie对一个回答中所有短句预测出的原因都重新放到对应于相应回答的list中

# rank_data_process.py
可选参数 --usage
--usage train : 生成用于训练，测试排序模型的数据
--usage predict ：生成待预测排序的数据
也可以直接在其他模块调用函数，直接返回需要的以上数据

# train.py
训练排序模型

# test.py 
测试排序模型
可选参数 --top
--top top1 : 评估top1的准确率
--top top2 : 评估top2的准确率

# predict.py
给一个回答下至少预测出了两个原因的样本进行原因排序，并将排序后的原因列表写入字典

