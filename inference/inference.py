import sys
sys.path.append('/data/pzy2022/project/eccnlp')
# sys.path.append('/data/fkj2023/Project/eccnlp')
import config
from config import re_filter

from utils import read_list_file, split_dataset, evaluate_sentence

import logging
from info_extraction.inference import extraction_inference

from data_process.info_extraction import dataset_generate_train

from phrase_rerank.rank_data_process import get_logger1,  form_predict_input_list, add_embedding, get_text_list, merge_reasons,read_word
from phrase_rerank.lambdarank import LambdaRank, add_rerank, predict
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def classification():
    return

def extraction(args, dataset):
    result = extraction_inference(args, dataset)
    return result

def rerank_predict(args, uie_list):
    word = read_word(args.word_path)
    after_embedding_list = add_embedding(args, uie_list)
    text_list, num_list = get_text_list(uie_list)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)
    predict_list, reasons = form_predict_input_list(args, merged_list, word)
    predict_data = np.array(predict_list)
    predicted_list, rerank_reasons, rerank_scores = predict(args, predict_data, reasons)
    logpath4 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/predict/" 
    log4 = get_logger1('inference_add_rerank',logpath4)
    res = add_rerank(args, rerank_reasons,rerank_scores, merged_list, log4)
    return res



if __name__ == '__main__':
    args = config.get_arguments()

    dataset = read_list_file(args.data)
    # clf = read_list_file('/data/xf2022/Projects/eccnlp_local/data/result_data/result_data2.0_TextRCNN2022_12_21_15_37.txt')

    logging.info(f'length of raw dataset: {len(dataset)}')

    dataset = re_filter(dataset)
    train_data, dev_data, test_data = dataset_generate_train(args, dataset)
    logging.info(f'{len(dataset)} samples left after re_filter')
    
    result = extraction(args, test_data)
    # # evaluate_sentence(result, clf)


    # phrase_rerank

    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    # uie_list = read_list_file(filepath)
    # rerank_res = rerank_predict(args, uie_list)
    rerank_res = rerank_predict(args, result)


