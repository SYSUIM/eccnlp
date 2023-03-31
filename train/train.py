import os
import sys
sys.path += '..'
import argparse
import logging
from multiprocessing import Process, Pool

from utils import read_list_file, evaluate_sentence, get_logger, check_log_dir, split_train_datasets, accuracy_k, reciprocal_rank_k, average_precision_k
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
from phrase_rerank.rank_data_process import form_input_list, add_embedding, get_text_list, merge_reasons, read_word, uie_list_filter, split_rerank_data, form_predict_input_list
from phrase_rerank.lambdarank import LambdaRank, train_rerank, validate_rerank, precision_k, predict_rank, add_rerank
from datetime import datetime
import numpy as np
import torch
from utils import read_list_file, check_log_dir
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


def evaluate_model(model_name, model_res_list):
    k_values = [1, 2, 3, 4, 5, 20]
    metrics = [('accuracy', accuracy_k), ('MRR', reciprocal_rank_k), ('MAP', average_precision_k)]

    for metric_name, metric_func in metrics:
        for k in k_values:        
            scores = [metric_func(data, model_name, args.type, k) for data in model_res_list]
            logging.info(f'{metric_name}@{k} on filted_test_data_uie_res: {np.mean(scores)}')



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
    train_data, dev_data, test_data = dataset_generate_train(args.train_size, args.val_size, data)
    logging.info(f'train_data: {len(train_data)}, dev_data: {len(dev_data)}, test_data: {len(test_data)}')

    do_train(device = args.device,
             seed = args.seed,
             model_dir = args.model_dir,
             UIE_model = args.UIE_model,
             UIE_batch_size = args.UIE_batch_size,
             max_seq_len = args.max_seq_len,
             init_from_ckpt = args.init_from_ckpt,
             UIE_learning_rate = args.UIE_learning_rate,
             UIE_num_epochs = args.UIE_num_epochs,
             logging_steps = args.logging_steps,
             valid_steps = args.valid_steps,
             save_dir = args.save_dir,
             train_data = train_data,
             dev_data = dev_data)
    
    result_on_train_data, result_on_dev_data, result_on_test_data = extraction_inference(train_data, dev_data, test_data, args.type, args.save_dir, args.position_prob)
    
    filted_result_on_test_data = [data for data in result_on_test_data if len(data['output'][0]) != 0]
    logging.info(f'length of filted_result_on_test_data: {len(filted_result_on_test_data)}')

    evaluate_model('uie', filted_result_on_test_data)

    return result_on_train_data, result_on_dev_data, result_on_test_data, filted_result_on_test_data



def run_rerank(args, uie_list, word, filted_result_on_test_data):

    #embedding
    # after_embedding_list = add_embedding(args, uie_list)
    filtered_uie_list_train, context_list, filtered_uie_list_predict = uie_list_filter(args, uie_list)
    after_embedding_list = add_embedding(args, filtered_uie_list_train)

    # #merge reasons
    text_list, num_list = get_text_list(filtered_uie_list_train)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)
    train_merged_list, test_merged_list = split_rerank_data(merged_list)

    logging.info(f'merged_list length: {len(merged_list)}')
    logging.info(f'train_merged_list length: {len(train_merged_list)}')
    logging.info(f'test_merged_list length: {len(test_merged_list)}')

    #train
    train_list, reason_of_train = form_input_list(args, train_merged_list, word)
    training_data = np.array(train_list)
    model = LambdaRank(training_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_rerank(args, training_data, device, model)


    # evaluate
    test_list, test_reason = form_predict_input_list(args, test_merged_list, word)
    test_data = np.array(test_list)
    ndcg , pred_scores= validate_rerank(args, test_data, 2)
     
    # compare
    logging.info(f'compare rerank_res with uie_res on filted_result_on_uie_test_data...')
    logging.info(f'length of filted_result_on_test_data: {len(filted_result_on_test_data)}')
    filted_result_on_uie_test_data = add_embedding(args, filted_result_on_test_data)
    test_list_uie, test_reason_uie = form_predict_input_list(args, filted_result_on_uie_test_data, word)
    test_data_uie = np.array(test_list_uie)
    predicted_list, rerank_reasons, rerank_scores = predict_rank(args, test_data_uie, test_reason_uie)
    rerank_res_on_uie_filted_test_data = add_rerank(args, rerank_reasons, rerank_scores, filted_result_on_uie_test_data)
    logging.info(f'length of rerank_res_on_uie_filted_test_data: {len(rerank_res_on_uie_filted_test_data)}')
    
    evaluate_model('rerank', rerank_res_on_uie_filted_test_data)
    

    # precision_k(args, test_data)

    return 


if __name__ == '__main__':
    args = config.get_arguments()


    log_path = check_log_dir(args.time)

    # main_logger = get_logger('main_logger', log_path + '/main.log')

    args_message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logging.info(f'\n{args_message}')

    # clf_logger = get_logger('clf_logger', log_path + '/clf.log')
    # ext_logger = get_logger('ext_logger', log_path + '/ext.log')

    raw_dataset = read_list_file(args.data)
    logging.info(f'length of raw dataset: {len(raw_dataset)}')


    # waiting for re filter...
    # dataset = re_filter(raw_dataset)
    # main_logger.info(f'{len(raw_dataset) - len(dataset)} samples are filted by re_filter')
    # res = ensemble_text_classification(args, dataset)
    # for i in res:
    #     print(i)
    # exit(0)

    report, matrix = BertForClassification(args, raw_dataset)
    print(report, matrix)


    result_on_train_data, result_on_dev_data, result_on_test_data, filted_result_on_test_data = run_information_extraction(args, raw_dataset)
    uie_list = result_on_train_data + result_on_dev_data + result_on_test_data

    # run_rerank
    word = build_thesaurus(raw_dataset, args.t_path)
    run_rerank(args, uie_list, word, filted_result_on_test_data)
    # exit(0)

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

