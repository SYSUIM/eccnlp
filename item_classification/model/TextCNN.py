# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

now_time = time.strftime('%m.%d_%H.%M', time.localtime())

class Config(object):

    """配置参数"""
    def __init__(self, train_list, dev_list, test_list, embeddings_pretrained, embedding, args):
        self.model_name = 'TextCNN'
        self.train_list = train_list                                                        # 训练集
        self.dev_list = dev_list                                                            # 验证集
        self.test_list = test_list                                                          # 测试集
        self.class_list = [x.strip() for x in open(
            './data/class.txt', encoding='utf-8').readlines()]                         # 类别名单
        self.save_path = './data/save_model/' + self.model_name + now_time + '.ckpt'   # 模型训练结果
        self.embedding_pretrained = torch.tensor(
            embeddings_pretrained.astype('float32'))\
            if embedding != 'random' else None                                          # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # 设备
 
        self.dropout = 0.5                                                              # 随机失活
        self.require_improvement = args.require_improvement                             # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                                         # 类别数
        self.n_vocab = args.n_vocab                                                     # 词表大小，在运行时赋值
        self.num_epochs = args.num_epochs                                               # epoch数
        self.batch_size = args.batch_size                                               # mini-batch大小
        self.pad_size = args.pad_size                                                   # 每句话处理成的长度(短填长切)
        self.learning_rate = args.clf_learning_rate                                         # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300                           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                                   # 卷积核尺寸
        self.num_filters = 256                                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
