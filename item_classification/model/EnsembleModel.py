# coding: UTF-8
import torch


class Config(object):

    """配置参数"""
    def __init__(self, train_list, dev_list, test_list, embeddings_pretrained, embedding, args):
        self.model_name = 'EnsembleModel'
        self.train_list = train_list                                                        # 训练集
        self.dev_list = dev_list                                                            # 验证集
        self.test_list = test_list                                                          # 测试集
        self.embedding_pretrained = torch.tensor(
            embeddings_pretrained.astype('float32'))\
            if embedding != 'random' else None                                          # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # 设备


        self.batch_size = args.batch_size                                               # mini-batch大小
        self.pad_size = args.pad_size                                                   # 每句话处理成的长度(短填长切)
