# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
# from preprocess_re import get_dataset
import argparse
import pandas as pd
import logging

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--num_epochs', default=20, type=int, help='Total number of training epochs to perform.')
parser.add_argument('--require_improvement', default=2000, type=int, help='Stop the train if the improvement is not required.')
parser.add_argument('--n_vocab', default=0, type=int, help='Size of the vocab.')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU/CPU for training.')
parser.add_argument('--pad_size', default=32, type=int, help='Size of the pad.')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='The initial learning rate for Adam.')
args = parser.parse_args()

def run_classification_model(train_df, dev_df, test_df):
    log_filename = '../data/log/' + args.model + time.strftime('%m.%d_%H.%M', time.localtime()) + '.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

    # 回答文本:pre_trained, 随机初始化:random
    embedding = 'pre_trained'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'Transformer':
        args.learning_rate = 5e-4
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif, get_utils
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif, get_utils

    embeddings_pretrained = get_utils(train_df)

    x = import_module('model.' + model_name)
    config = x.Config(train_df, dev_df, test_df, embeddings_pretrained, embedding, args)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    logging.info("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    logging.info("Time usage:{}".format(time_dif))

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    logging.info(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)

# 测试run_classification_model方法
if __name__ == '__main__':
    # train_df, dev_df, test_df = get_dataset(data, args)
    train_df = pd.read_csv('../data/dataset/train.txt', sep='\t', names=['content', 'label'], encoding='utf-8')
    dev_df = pd.read_csv('../data/dataset/validation.txt', sep='\t', names=['content', 'label'], encoding='utf-8')
    test_df = pd.read_csv('../data/dataset/test.txt', sep='\t', names=['content', 'label'], encoding='utf-8')
    run_classification_model(train_df, dev_df, test_df)
