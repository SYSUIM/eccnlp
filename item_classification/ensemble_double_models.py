# coding: UTF-8
import time
import torch
import numpy as np
from item_classification.soft_ensemble_train_eval import *
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

now_time = time.strftime('%m.%d_', time.localtime())

def ensemble_double_models(train_lists, dev_lists, test_list, args):
    # log_filename = './data/log/' + args.model + time.strftime('%m.%d_%H.%M', time.localtime()) + '.log'
    # logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

    ensemble_num = len(train_lists)

    # 回答文本:pre_trained, 随机初始化:random
    embedding = 'random'
    if args.embedding == 'random':
        embedding = 'random'

    # models = ['TextCNN', 'TextRNN', 'TextRCNN', 'TextRNN_Att', 'DPCNN']
    raw_labels = []     #原标签
    predict_models_labels = []   #所有模型的预测值
    predict_pr_positive = []   # 所有模型预测正类的概率


    def ensemble_single_model(n, model_name, args):
        # for i, model_name in enumerate(models):
        for i in range(ensemble_num):
            logging.info('-'*30 + str(n) + '_' + str(i) + ' ' + model_name + '-'*30)
            # print(len(train_datasets[i]))
            filename_trimmed_dir = "/data/xf2022/Projects/eccnlp_local/data/embedding/" + "embedding_answer_" + model_name + str(i)
            embeddings_pretrained = get_utils(train_lists[i], filename_trimmed_dir)
            embedding = 'embedding_answer_' + model_name + str(i) + '.npz'
            x = import_module('item_classification.model.' + model_name)
            config = x.Config(train_lists[i], dev_lists[i], test_list, embedding, args)
            config.save_path = './data/save_model/' + model_name + now_time + str(i) + '.pth'
            config.vocab_path = '/data/xf2022/Projects/eccnlp_local/data/vocab/vocab_' + model_name + str(i) + '.pkl'
            np.random.seed(args.classification_model_seed)
            torch.manual_seed(args.classification_model_seed)
            torch.cuda.manual_seed_all(args.classification_model_seed)
            torch.backends.cudnn.deterministic = True  # 保证每次结果一样

            start_time = time.time()
            logging.info("Loading data...")
            vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
            train_iter = build_iterator(train_data, config)
            # print(train_iter)
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
            labels_all, predict_all, loss_total, predict_pr_all = get_predict_label(config, model, test_iter)
            logging.info(f"predict_pr_all type:{type(predict_pr_all)}")
            logging.info(f"predict_pr_all shape:{predict_pr_all.shape}")
            predict_pr_positive.append(predict_pr_all.reshape(-1, 2)[:, -1])
            raw_labels.append(labels_all)
            # predict_labels.append(predict_all)
        
        predict_pr_positive_np = np.array(predict_pr_positive, dtype=float)

        # soft voting
        predict_labels = np.array(np.rint(predict_pr_positive_np.mean(axis=0)), dtype=int)
        
        return predict_labels, labels_all

    for n, model_name in enumerate(args.ensemble_models):
        predict_labels, labels_all = ensemble_single_model(n, model_name, args)
        predict_models_labels.append(predict_labels)
        predict_labels = np.array(predict_labels, dtype=int)
        labels_all = np.array(labels_all, dtype=int)
        logging.info('-'*30 + str(n) + 'Single EnsembleModels' + '-'*30)
        evaluate(labels_all, predict_labels, test=True)
        
    predict_models_labels = np.array(predict_models_labels, dtype=int)
    raw_labels = np.array(raw_labels[0], dtype=int)

    # hard voting
    predict_models_labels = np.array(np.rint(predict_models_labels.mean(axis=0)), dtype=int)
        

    # evalute
    logging.info('-'*30 + 'Double EnsembleModels' + '-'*30)
    evaluate(raw_labels, predict_models_labels, test=True)

    return predict_models_labels    



    # predict
    # predict_data = [i for i in test_data]
    # predict_iter = build_iterator(predict_data, config)

    # def predict(config, model, predict_iter):
    #     model.load_state_dict(torch.load(config.save_path))
    #     model.eval()
    #     predict_all = np.array([], dtype=int)
    #     with torch.no_grad():
    #         for texts, labels in predict_iter:
    #             outputs = model(texts)
    #             predic = torch.max(outputs.data, 1)[1].cpu().numpy()
    #             predict_all = np.append(predict_all, predic)

    #     return predict_all
    
    # predict_all = predict(config, model, predict_iter)

    # return predict_all