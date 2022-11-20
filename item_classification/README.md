# 归因文本分类-Pytorch
基于Pytorch的业绩归因文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention，DPCNN，Transformer

模型参考：https://github.com/649453932/Chinese-Text-Classification-Pytorch

## 介绍
业绩说明会问答文本，数据以字为输入单位，预训练词向量使用[Financial News 金融新闻](https://github.com/Embedding/Chinese-Word-Vectors)

## 环境
- python 3.7
- pytorch 1.1
- tqdm
- sklearn
- tensorboardX

## 数据预处理
1. 将非业绩归因回答进行正则过滤，去除部分标注为0的样本。
2. 对回答文本进行切分，对于原句的【是否为业绩归因回答】标注为1的子句，如果原因归属在子句中，则该子句标注为1，否则标注为0；对于原句【是否为业绩归因回答】标注为0的子句，保持标注为0。
3. 采用上采样均衡训练样本，生成训练集、验证集和测试集。 
__数据集划分:__

| 数据集 | 数据量 |
| :---: | :----: |
| 训练集 |        |
| 验证集 |       |
| 测试集 |       |

## 分类：是否为业绩归因回答
预训练词向量：utils.py生成词表并提取词表对应的预训练词向量。  
__效果__
| 模型 | acc | precision | F1-score | 备注 |
|:----:|:---:|:---------:|:--------:|:----:|
|      |     |           |          |      |
|      |     |           |          |      |
|      |     |           |          |      |
|      |     |           |          |      |
|      |     |           |          |      |
|      |     |           |          |      |
|      |     |           |          |      |
