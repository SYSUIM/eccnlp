# coding: UTF-8
import time
import torch
import numpy as np
from item_classification.ensemble_train_eval import *
from item_classification.utils import build_dataset, build_iterator, get_time_dif, get_utils
from importlib import import_module
import argparse
import pandas as pd
import logging

# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--data', type=str, required=True, help='path of raw data')
# parser.add_argument('--min_length', default=5, type=str, help='not less than 0')
# parser.add_argument('--type', default='业绩归因', type=str, help='type of answer')
# parser.add_argument('--balance', default='none',type=str, required=True, help='up or down or none')
# parser.add_argument('--log', type=str, required=True, help='path of log file')
# parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# parser.add_argument('--num_epochs', default=20, type=int, help='Total number of training epochs to perform.')
# parser.add_argument('--require_improvement', default=2000, type=int, help='Stop the train if the improvement is not required.')
# parser.add_argument('--n_vocab', default=0, type=int, help='Size of the vocab.')
# parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU/CPU for training.')
# parser.add_argument('--pad_size', default=32, type=int, help='Size of the pad.')
# parser.add_argument('--learning_rate', default=1e-3, type=float, help='The initial learning rate for Adam.')
# args = parser.parse_args()

def ensemble_classification_model(train_list, dev_list, test_list, args):
    # log_filename = './data/log/' + args.model + time.strftime('%m.%d_%H.%M', time.localtime()) + '.log'
    # logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

    # 回答文本:pre_trained, 随机初始化:random
    embedding = 'pre_trained'
    if args.embedding == 'random':
        embedding = 'random'

    embeddings_pretrained = get_utils(train_list)

    # models = ['TextCNN', 'TextRNN', 'TextRCNN', 'TextRNN_Att', 'DPCNN']
    raw_labels = []     #原标签
    predict_labels = []   #所有模型的预测值

    x = import_module('item_classification.model.' + args.model)
    config = x.Config(train_list, dev_list, test_list, embeddings_pretrained, embedding, args)
    np.random.seed(args.classification_model_seed)
    torch.manual_seed(args.classification_model_seed)
    torch.cuda.manual_seed_all(args.classification_model_seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    logging.info("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    logging.info("Time usage:{}".format(time_dif))

    for i, model_name in enumerate(args.ensemble_models):
        logging.info('-'*30 + str(i) + ' ' + model_name + '-'*30)
        x = import_module('item_classification.model.' + model_name)
        config = x.Config(train_list, dev_list, test_list, embeddings_pretrained, embedding, args)

        # train
        config.n_vocab = len(vocab)
        model = x.Model(config).to(config.device)
        if model_name != 'Transformer':
            init_network(model)
        logging.info(model.parameters)
        train(config, model, train_iter, dev_iter, test_iter)
        labels_all, predict_all, loss_total = get_predict_label(config, model, test_iter)
        raw_labels.append(labels_all)
        predict_labels.append(predict_all)
    
    raw_labels = np.array(raw_labels[0], dtype=int)
    predict_labels = np.array(predict_labels, dtype=int)

    #voting
    predict_labels = np.array(np.rint(predict_labels.mean(axis=0)), dtype=int)


    #evalute
    logging.info('-'*30 + 'EnsembleModels' + '-'*30)
    evaluate(raw_labels, predict_labels, test=True)    

    # return predict_labels