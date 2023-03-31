import os
import config

import sys
sys.path.append('/data/fkj2023/Project/eccnlp')
from config import re_filter
from utils import read_list_file, split_dataset, evaluate_sentence, read_word, print_list, check_log_dir, get_res_logger
from data_process.dataprocess import train_dataset, split_dataset, classification_dataset, dict_to_list

import logging

from item_classification.bert_inference import bertForSequenceClassification

from item_classification.classification_model_predict import classification_models_predict

from info_extraction.inference import extraction_inference

# from data_process.info_extraction import dataset_generate_train

from phrase_rerank.rank_data_process import form_predict_input_list, add_embedding, get_text_list, merge_reasons, uie_list_filter
from phrase_rerank.lambdarank import LambdaRank, add_rerank, predict_rank
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def bertFilter(args, dataset):
    filtedDataset = bertForSequenceClassification(args, dataset)

    return filtedDataset

def classification(args, dataset):
    predict_list = dict_to_list(dataset)
    predict_all = classification_models_predict(predict_list, args)
    logging.info(f'all predict dict nums:{len(predict_all)}')

    for i in range(len(predict_all)):
        dataset[i]['label'] = predict_all[i]


    # with open("./data/result_data/3.1_result_dict_DoubleEnsemble0309.txt", 'w', encoding='utf8') as f:
    #     for i in range(len(dataset)):
    #         f.write(str(dataset[i]) + '\n')

    return dataset

def extraction(args, dataset):
    _, _, result = extraction_inference(None, None, dataset, args.type, args.save_dir, args.position_prob)
    return result

def rerank_predict(args, uie_list):

    word = read_word(args.word_path)
    filtered_uie_list_train, context_list, filtered_uie_list_preict = uie_list_filter(args, uie_list)
    after_embedding_list = add_embedding(args, filtered_uie_list_preict)
    text_list, num_list = get_text_list(filtered_uie_list_preict)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)
    predict_list, reasons = form_predict_input_list(args, merged_list, word)
    predict_data = np.array(predict_list)
    predicted_list, rerank_reasons, rerank_scores = predict_rank(args, predict_data, reasons)
    res = add_rerank(args, rerank_reasons, rerank_scores, merged_list)
    logging.info(f'length of rerank_res: {len(res)}')
    return res



if __name__ == '__main__':
    args = config.get_arguments()

    log_path = check_log_dir(args.time) 
    # print(log_path)
    # exit(0)
 
    # read predict data...
    # predict_dataset = read_list_file(args.predict_data)
    # logging.info(f'length of raw dataset: {len(predict_dataset)}')

    # # waiting for re filter...
    # dataset = re_filter(predict_dataset)
    # logging.info(f'{len(predict_dataset) - len(dataset)} samples are filted by re_filter')
    # logging.info(f'all dataset dict nums:{len(dataset)}')
    
    # dataset = classification(args, dataset)
    # logging.info('double ensemble predict completed.')


    dataset = read_list_file(args.data)
    # dataset = read_list_file('/data/xf2022/Projects/eccnlp_local/data/result_data/3.1_result_dict_predict20.46.txt')

    logging.info(f'length of raw dataset: {len(dataset)}')

    # dataset = re_filter(dataset)

    filtedDataset = bertFilter(args, dataset)
    with open(log_path +"/2.2_uni_bertfiltedDataset.txt", 'w') as f:
        [f.write(str(data) + '\n') for data in filtedDataset]    

    # train_data, dev_data, test_data = dataset_generate_train(args, dataset)
    # logging.info(f'{len(dataset)} samples left after re_filter')
    
    result = extraction(args, filtedDataset)
    # with open("./after_extraction_data3.1DoubleEnsemble.txt", 'w') as f:
    #     [f.write(str(data) + '\n') for data in result]
    # # evaluate_sentence(result, clf)
    with open(log_path +"/2.2_uni_uie_res.txt", 'w') as f:
        [f.write(str(data) + '\n') for data in result ]  

    # phrase_rerank

    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    # uie_list = read_list_file(filepath)
    # rerank_res = rerank_predict(args, uie_list)
    rerank_res = rerank_predict(args, result)   
    
    with open(log_path +"/2.2_uni_add_rerank.txt", 'w') as f:
        [f.write(str(data) + '\n') for data in rerank_res]    


