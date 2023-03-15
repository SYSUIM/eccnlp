import os
import sys
sys.path += '..'
import argparse
import logging
from multiprocessing import Process, Pool

from utils import read_list_file, evaluate_sentence, get_logger, check_log_dir, split_train_datasets
import config
from config import re_filter

# data preprocess
from data_process.dataprocess import re_pattern1, re_pattern2, train_dataset, split_dataset, classification_dataset, dict_to_list

# item classification
from item_classification.run_classification_model import run_classification_model
from item_classification.ensemble_models import ensemble_classification_model
from item_classification.ensemble_double_models import ensemble_double_models
from item_classification.bert_train import BertForClassification

# information extraction
from info_extraction.finetune import do_train
from info_extraction.inference import extraction_inference
from data_process.info_extraction import dataset_generate_train


# phrase rerank
from phrase_rerank.rank_data_process import get_logger1, get_logger2, form_input_list, print_list, add_embedding, get_text_list, merge_reasons, read_word
from phrase_rerank.lambdarank import LambdaRank, train, validate, precision_k
from datetime import datetime
import numpy as np
import torch
from config import re_filter
from utils import read_list_file, get_logger, check_log_dir
import config
from data_process.dataprocess import build_thesaurus


# def get_logger(log_name, log_file, level = logging.INFO):
#     formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s')
#     # logging.basicConfig(
#     #     level = level,
#     #     format = '%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
#     #     datefmt = '%Y-%m-%d %H:%M:%S'
#     # )
    
#     logger = logging.getLogger(log_name)
#     fileHandler = logging.FileHandler(log_file, mode='a')
#     fileHandler.setFormatter(formatter)
    
#     logger.setLevel(level)
#     logger.addHandler(fileHandler)

#     return logger


def text_classification(args, data):
    data = re_pattern1(args)
    dataset = train_dataset(data, args)

    # p_data = re_pattern11(args)
    # p_dataset = predict_dataset(p_data, args)
    # predict_list = dict_to_list(p_dataset)

    train_dict,val_dict,test_dict = split_dataset(dataset,args)
    train_list, dev_list, test_list, train_dict, val_dict, test_dict = classification_dataset(train_dict,val_dict,test_dict, args)
    # predict_test = run_classification_model(train_list, dev_list, predict_list, args)
    # for i in range(len(predict_test)):
    #     test_dict[i]['label'] = predict_test[i]
    # all_dict = []
    # all_dict.extend(train_dict)
    # all_dict.extend(val_dict)
    # all_dict.extend(test_dict)
    # return all_dict
    # for i in range(len(predict_test)):
    #     p_dataset[i]['label'] = predict_test[i]
    # # with open("./data/result_data/result_data_TextRNN.txt", 'w', encoding='utf8') as f:
    # #     for i in range(len(all_dict)):
    # #         f.write(str(all_dict[i]) + '\n')
    # return p_dataset


def ensemble_text_classification(args, dataset):
    # dataset = train_dataset(data, args)
    train_dict, val_dict, test_dict = split_dataset(dataset, args)
    train_list, dev_list, test_list, train_dict, val_dict, test_dict = classification_dataset(train_dict,val_dict,test_dict, args)
    predict_test = ensemble_classification_model(train_list, dev_list, test_list, args)
    # ensemble_single_model(train_list, dev_list, test_list, args)
    for i in range(len(predict_test)):
        test_dict[i]['label'] = predict_test[i]
    all_dict = []
    # all_dict.extend(train_dict)
    # all_dict.extend(val_dict)
    all_dict.extend(test_dict)
    # with open("./data/result_data/result_data_TextRNN.txt", 'w', encoding='utf8') as f:
    #     for i in range(len(all_dict)):
    #         f.write(str(all_dict[i]) + '\n')
    return all_dict


def ensemble_double_classifications(args, dataset):
    train_dict, val_dict, test_dict = split_dataset(dataset,args)
    logging.info("split all dataset completed.")
    # test_dict = read_list_file(args.predict_data)
    # logging.info(f'length of raw dataset: {len(test_dict)}')
    train_datasets = split_train_datasets(train_dict, args)
    logging.info("split train dataset completed.")
    dev_datasets = split_train_datasets(val_dict, args)
    logging.info("split dev dataset completed.")
    train_lists, dev_lists = [], []
    for i in range(len(train_datasets)):
        train_lists.append(dict_to_list(train_datasets[i]))
        dev_lists.append(dict_to_list(dev_datasets[i]))
    logging.info("train and dev datasets into lists completed.")
    test_list = dict_to_list(test_dict)
    logging.info("test dataset into list completed.")

    predict_test = ensemble_double_models(train_lists, dev_lists, test_list, args)

    for i in range(len(predict_test)):
        test_dict[i]['label'] = predict_test[i]
    all_dict = []
    all_dict.extend(train_dict)
    all_dict.extend(val_dict)
    all_dict.extend(test_dict)

    # with open("./data/result_data/2.1_result_dict_DoubleEnsemble0308.txt", 'w', encoding='utf8') as f:
    #     for i in range(len(all_dict)):
    #         f.write(str(all_dict[i]) + '\n')

    return all_dict


def run_information_extraction(args, data):
    train_data, dev_data, test_data = dataset_generate_train(args, data)
    main_logger.info(f'train_data: {len(train_data)}, dev_data: {len(dev_data)}, test_data: {len(test_data)}')

    do_train(args, train_data, dev_data)
    
    result_on_train_data, result_on_dev_data, result_on_test_data = extraction_inference(args, train_data, dev_data, test_data)
    
    return result_on_train_data, result_on_dev_data, result_on_test_data



def run_rerank(args, uie_list, word):

    #embedding
    after_embedding_list = add_embedding(args, uie_list)
    logpath1 = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/embedding/'
    log1 = get_logger1("embedding_list",logpath1)
    print_list(after_embedding_list, log1)

    #merge reasons
    text_list, num_list = get_text_list(uie_list)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)

    #train
    logpath3 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/train_lambdarank/" 
    log3 = get_logger2('train_ndcg',logpath3)
    epoch = 300
    learning_rate = 0.0001
    all_list, train_list, test_list, reason_of_test = form_input_list(args, merged_list, word)
    training_data = np.array(train_list)
    model = LambdaRank(training_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    modelpath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_v0_parameter'+datetime.now().strftime("%m_%d_%H_%M_%S")+'.pkl'
    train(training_data, learning_rate, epoch, modelpath, device, model, log3)

    # value
    validate_data = np.array(test_list)
    k = 2
    ndcg , pred_scores= validate(validate_data, k, modelpath)
    # log3.info("pred_scores: %s", pred_scores)
    log3.info("np.nanmean(ndcg): %s", np.nanmean(ndcg))
    precision_k(validate_data, modelpath, log3)

    return 


if __name__ == '__main__':
    args = config.get_arguments()

    log_path = check_log_dir(args.time)

    main_logger = get_logger('main_logger', log_path + '/main.log')

    args_message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    main_logger.info(f'\n{args_message}')

    # clf_logger = get_logger('clf_logger', log_path + '/clf.log')
    # ext_logger = get_logger('ext_logger', log_path + '/ext.log')

    raw_dataset = read_list_file(args.data)
    main_logger.info(f'length of raw dataset: {len(raw_dataset)}')

    # waiting for re filter...
    # dataset = re_filter(raw_dataset)
    # main_logger.info(f'{len(raw_dataset) - len(dataset)} samples are filted by re_filter')
    # res = ensemble_text_classification(args, dataset)
    # for i in res:
    #     print(i)
    # exit(0)

    report, matrix = BertForClassification(args, raw_dataset)
    print(report, matrix)

    # all_dict = text_classification(args)
    # all_dict = ensemble_text_classification(args, dataset)
    # logging.info('ensemble_text_classification training completed.')
    
    # waiting for ensemble double models...
    # all_dict = ensemble_double_classifications(args, dataset)
    # logging.info(f"all dict nums:{len(all_dict)}")

    # test = run_information_extraction(args, dataset)
    # with open("./20220228.txt", 'w') as f:
    #     [f.write(str(data) + '\n') for data in test]
    # exit(0)

    # main_logger.info(f'parent pid: {os.getpid()}')
    # processes = [
    #     Process(target = ensemble_text_classification, args = (args, dataset)),
    #     # Process(target = ensemble_double_classifications, args = (args, dataset)),
    #     Process(target = run_information_extraction, args = (args, dataset))
    # ]

    # [p.start() for p in processes]
    # [main_logger.info(f'{p} pid is: {p.pid}') for p in processes]
    # [p.join() for p in processes]
    # classification_result, extraction_result = [p.get() for p in processes]

    # evaluate for sentences after extraction
    # evaluate_sentence(extraction_result, classification_result)
    pool = Pool(2)
    main_logger.info(f'parent pid: {os.getpid()}')
    result_list = []
    result_list.append(pool.apply_async(func = ensemble_double_classifications, args = (args, dataset)))
    result_list.append(pool.apply_async(func = run_information_extraction, args = (args, dataset)))
    pool.close()
    pool.join()
    classification_result = result_list[0].get()
    print("======================================classification_result======================================")
    # print(extraction_result)
    for i in classification_result:
        print(i)

    extraction_result = result_list[1].get()
    uie_list = []
    for i in extraction_result:
        uie_list = uie_list + i

    print("======================================extraction_result======================================")
    # print(extraction_result)
    for i in uie_list:
        print(i)
    print("===========================end==============")

    # run_rerank
    word = build_thesaurus(dataset, args.t_path)
    # uie 结果路径
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    # uie_list = read_list_file(filepath)
    # run_rerank(args, uie_list, word)  
    run_rerank(args, uie_list, word)

"""
    # 跑通了，...
    res = run_information_extraction(args, dataset)
    uie_list = []
    for i in res:
        uie_list = uie_list + i
    # run_rerank
    word = build_thesaurus(dataset, args.t_path)
    run_rerank(args, uie_list, word)
    exit(0)
"""
    
