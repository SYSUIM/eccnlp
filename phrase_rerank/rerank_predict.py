import argparse
from rank_data_process import get_logger1,  form_predict_input_list, read_list, print_list, add_embedding, get_text_list, merge_reasons, read_word
import numpy as np
from lambdarank import LambdaRank, add_rerank, predict, precision_k
import torch
from datetime import datetime


# def rerank_predict(args, uie_list):
def rerank_predict1(args):

    predict_start = datetime.now()
    word = read_word(args.word_path)

    #直接从合并好的文件开始执行
    filepath ='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/merged/2023-03-02_merged_list.log'
    merged_list = read_list(filepath)

    #predict   
    predict_list, reasons = form_predict_input_list(args, merged_list, word)
    predict_data = np.array(predict_list)
    predicted_list, rerank_reasons, rerank_scores = predict(args, predict_data, reasons)
    logpath4 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/predict/" 
    log4 = get_logger1('result_add_rerank',logpath4)
    res = add_rerank(args, rerank_reasons,rerank_scores, merged_list, log4)
    predict_end = datetime.now()
    log4.info("predict time : %s  minutes", (predict_end - predict_start).seconds/60)
    precision_k(predict_data, args.lambdarank_path, log4)
    log4.info("lambdarank model path: %s", args.lambdarank_path)
    return res 

def rerank_predict(args, uie_list):

    # 从最原始的uie结果开始执行
    predict_start = datetime.now()
    word = read_word(args.word_path)
    after_embedding_list = add_embedding(args, uie_list)
    text_list, num_list = get_text_list(uie_list)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)

    #predict   
    predict_list, reasons = form_predict_input_list(args, merged_list, word)
    predict_data = np.array(predict_list)
    predicted_list, rerank_reasons, rerank_scores = predict(args, predict_data, reasons)
    logpath4 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/predict/" 
    log4 = get_logger1('result_add_rerank',logpath4)
    res = add_rerank(args, rerank_reasons,rerank_scores, merged_list, log4)
    predict_end = datetime.now()
    log4.info("predict time : %s  minutes", (predict_end - predict_start).seconds/60)
    precision_k(predict_data, args.lambdarank_path, log4)
    log4.info("lambdarank model path: %s", args.lambdarank_path)
    return res 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='rerank_predict')
    parser.add_argument('--type', type=str, default='业绩归因',help='type of answer')
    parser.add_argument('--f_num', type=int, default=34, help='feature number')
    parser.add_argument('--vocab_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/bert_model/vocab.txt',help='vocab path')
    parser.add_argument('--code_length', type=int, default=16,help='the dimension of sentence features') 
    parser.add_argument('--lambdarank_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_v0_parameter_bak.pkl',help='lambdarank path')
    parser.add_argument('--word_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/2022-12-24_word.log',help='word path')
    args = parser.parse_args()

    # 从最原始的uie结果开始执行
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    # uie_list = read_list(filepath)
    # rerank_res = rerank_predict(args, uie_list)


    # 直接从合并好的文件开始执行
    rerank_res = rerank_predict1(args)

    # nohup python rerank_predict.py > rerank_predict.out &

