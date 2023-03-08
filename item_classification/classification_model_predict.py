# coding: UTF-8
import time
import torch
import numpy as np
# from item_classification.ensemble_train_eval import *
from item_classification.utils_predict import build_dataset, build_iterator, get_time_dif, get_utils
from importlib import import_module
import argparse
import pandas as pd
import logging
import os

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

def classification_models_predict(predict_list, args):
    # log_filename = './data/log/' + args.model + time.strftime('%m.%d_%H.%M', time.localtime()) + '.log'
    # logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # def force_cudnn_initialization():
    #     s = 32
    #     dev = torch.device('cuda:4')
    #     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    # force_cudnn_initialization()

    # 回答文本:pre_trained, 随机初始化:random
    embedding = 'random'
    if args.embedding == 'random':
        embedding = 'random'

    # embeddings_pretrained = np.asarray([], dtype='float32')
    # embeddings_pretrained = get_utils(predict_list)

    # models = ['TextCNN', 'TextRNN', 'TextRCNN', 'TextRNN_Att', 'DPCNN']
    train_list = []
    dev_list = []

    x = import_module('item_classification.model.' + args.model)
    config = x.Config(train_list, dev_list, predict_list, embedding, args)
    # config.device = torch.device('cpu')
    np.random.seed(args.classification_model_seed)
    torch.manual_seed(args.classification_model_seed)
    torch.cuda.manual_seed_all(args.classification_model_seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    logging.info("Loading data...")
    vocab, predict_data = build_dataset(config, args.word)
    # print(len(predict_data))
    predict_iter = build_iterator(predict_data, config)
    # print(len(predict_iter))
    time_dif = get_time_dif(start_time)
    logging.info("Time usage:{}".format(time_dif))

    args.n_vocab = len(vocab)
    date = args.ensemble_date
    predict_pr_positive = []
    predict_models_labels = []
    for n, model_name in enumerate(args.ensemble_models):
        for i in range(args.num_ensemble):
            # load model
            embedding = 'embedding_answer_' + model_name + str(i) + '.npz'
            x = import_module('item_classification.model.' + model_name)
            config = x.Config(train_list, dev_list, predict_list, embedding, args)
            config.vocab_path = '/data/xf2022/Projects/eccnlp_local/data/vocab/vocab_' + model_name + str(i) + '.pkl'
            start_time = time.time()
            logging.info("Loading data...")
            vocab, predict_data = build_dataset(config, args.word)
            # print(len(predict_data))
            predict_iter = build_iterator(predict_data, config)
            # print(len(predict_iter))
            time_dif = get_time_dif(start_time)
            logging.info("Time usage:{}".format(time_dif))
            config.n_vocab = len(vocab)
            # config.device = torch.device('cpu')
            model = x.Model(config).to(config.device)
            model.load_state_dict(torch.load('/data/xf2022/Projects/eccnlp_local/data/save_model/' + model_name + date + '_' + str(i) + '.pth'))
            # predict
            model.eval()
            predict_pr_all = np.array([], dtype=float)
            with torch.no_grad():
                for texts, labels in predict_iter:
                    outputs = model(texts)
                    # logging.info(f"model output results:{outputs[:5]}")
                    predic_pr = torch.softmax(outputs.data, 1).cpu().numpy()
                    # logging.info(f"model predict probability result:{predic_pr[:5]}")
                    # if i == 1: logging.info(f"{model_name} model predict pr result:{predic_pr}")
                    predict_pr_all = np.append(predict_pr_all, predic_pr)

            # if i == 1: logging.info(model_name + "model predict pr all result:" + predict_pr_all)
            predict_pr_positive.append(predict_pr_all.reshape(-1, 2)[:, -1])
            # if i == 1: logging.info(model_name + "model predict pr positive result:" + predict_pr_positive)

        predict_pr_positive_np = np.array(predict_pr_positive, dtype=float)
        # logging.info(f"{model_name} model predict positive result:{predict_pr_positive_np}")

        # soft voting
        predict_labels = np.array(np.rint(predict_pr_positive_np.mean(axis=0)), dtype=int)
        # logging.info(f"{model_name} model predict result:{predict_labels}")

        predict_models_labels.append(predict_labels)

    predict_models_labels = np.array(predict_models_labels, dtype=int)

    # hard voting
    predict_models_labels = np.array(np.rint(predict_models_labels.mean(axis=0)), dtype=int)
    logging.info(f"double ensemble models predict result:{predict_models_labels}")

    return predict_models_labels