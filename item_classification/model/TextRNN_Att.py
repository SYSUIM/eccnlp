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
        self.model_name = 'TextRNN_Att'
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
        self.learning_rate = args.learning_rate                                         # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300                           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                                          # lstm隐藏层
        self.num_layers = 2                                                             # lstm层数
        self.hidden_size2 = 64


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
