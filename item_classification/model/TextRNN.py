# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import time


now_time = time.strftime('%m.%d_%H.%M', time.localtime())

class Config(object):

    """配置参数"""
    def __init__(self, train_list, dev_list, test_list, embedding, args):
        self.model_name = 'TextRNN'
        self.train_list = train_list                                                        # 训练集
        self.dev_list = dev_list                                                            # 验证集
        self.test_list = test_list                                                          # 测试集
        self.class_list = [x.strip() for x in open(
            '/data/xf2022/Projects/eccnlp_local/data/class.txt', encoding='utf-8').readlines()]                         # 类别名单
        self.vocab_path = '/data/xf2022/Projects/eccnlp_local/data/vocab.pkl'
        self.save_path = './data/save_model/' + self.model_name + now_time + '.ckpt'   # 模型训练结果
        # self.embedding_pretrained = torch.tensor(
        #     embeddings_pretrained.astype('float32'))\
        #     if embedding != 'random' else None                                          # 预训练词向量
        self.embedding_pretrained = torch.tensor(
            np.load('/data/xf2022/Projects/eccnlp_local/data/embedding/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     # 设备

        self.dropout = 0.5                                                              # 随机失活
        self.require_improvement = args.require_improvement                             # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                                         # 类别数
        self.n_vocab = args.n_vocab                                                     # 词表大小，在运行时赋值
        self.num_epochs = args.num_epochs                                               # epoch数
        self.batch_size = args.batch_size                                               # mini-batch大小
        self.pad_size = args.pad_size                                                   # 每句话处理成的长度(短填长切)
        self.learning_rate = args.clf_learning_rate                                         # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300                           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                                          # lstm隐藏层
        self.num_layers = 2                                                             # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
