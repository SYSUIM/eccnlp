import os
import config
args = config.get_arguments()

if args.device[-1].isdigit():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devive[-1]
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('/data/pzy2022/project/eccnlp')
# sys.path.append('/data/fkj2023/Project/eccnlp')

from config import re_filter
from utils import read_list_file, split_dataset, evaluate_sentence

import logging

from item_classification.bert_inference import bertForSequenceClassification

from info_extraction.inference import extraction_inference

from data_process.info_extraction import dataset_generate_train

from phrase_rerank.rank_data_process import get_logger1,  form_predict_input_list, add_embedding, get_text_list, merge_reasons,read_word, print_list
from phrase_rerank.lambdarank import LambdaRank, add_rerank, predict
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def bertFilter(args, dataset):
    filtedDataset = bertForSequenceClassification(args, dataset)

    return filtedDataset

def classification():
    return

def extraction(args, dataset):
    result = extraction_inference(args, dataset)
    return result

def rerank_predict(args, uie_list):
    logpath4 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/inference/" 
    log1 = get_logger1('embedding',logpath4)
    log2 = get_logger1('merge',logpath4)
    log3 = get_logger1('inference_add_rerank',logpath4)

    word = read_word(args.word_path)
    after_embedding_list = add_embedding(args, uie_list)
    print_list(after_embedding_list, log1)
    text_list, num_list = get_text_list(uie_list)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)
    print_list(merged_list, log2)
    predict_list, reasons = form_predict_input_list(args, merged_list, word)
    predict_data = np.array(predict_list)
    predicted_list, rerank_reasons, rerank_scores = predict(args, predict_data, reasons)
    res = add_rerank(args, rerank_reasons,rerank_scores, merged_list, log3)
    return res



if __name__ == '__main__':
    args = config.get_arguments()
    

    # dataset = read_list_file(args.data)
    dataset = read_list_file('/data/xf2022/Projects/eccnlp_local/data/result_data/3.1_result_dict_predict20.46.txt')

    logging.info(f'length of raw dataset: {len(dataset)}')

    dataset = re_filter(dataset)

    filtedDataset = bertFilter(args, dataset)

    # train_data, dev_data, test_data = dataset_generate_train(args, dataset)
    # logging.info(f'{len(dataset)} samples left after re_filter')
    
    result = extraction(None, filtedDataset)
    with open("./after_extraction_data3.1DoubleEnsemble.txt", 'w') as f:
        [f.write(str(data) + '\n') for data in result]
    # # evaluate_sentence(result, clf)


    # phrase_rerank

    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    # uie_list = read_list_file(filepath)
    # rerank_res = rerank_predict(args, uie_list)
    # rerank_res = rerank_predict(args, result)


