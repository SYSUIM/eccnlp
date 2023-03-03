import argparse
from rank_data_process import get_logger1,  form_predict_input_list, read_list, print_list, add_embedding, get_text_list, merge_reasons, read_word
import numpy as np
from lambdarank import LambdaRank, add_rerank, predict, precision_k
import torch
from datetime import datetime


def rerank_predict1(args):

    predict_start = datetime.now()
    word = read_word(args.word_path)

    # begin with merged list
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

    # begin with uie result
    predict_start = datetime.now()
    word = read_word(args.word_path)

    #embedding
    embedding_start = datetime.now()
    after_embedding_list = add_embedding(args, uie_list)
    logpath1 = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/embedding/'
    log1 = get_logger1("pre_embedding_list",logpath1)
    print_list(after_embedding_list, log1)
    embedding_end = datetime.now()
    log1.info("embedding time : %s  minutes", (embedding_end - embedding_start).seconds/60 )

    #merge reasons
    merge_start = datetime.now()
    text_list, num_list = get_text_list(uie_list)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)
    logpath2 = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/merged/'
    log2 = get_logger1("pre_merged_list",logpath2)
    print_list(merged_list, log2)
    merge_end = datetime.now()
    log2.info("merge time : %s  minutes", (merge_end - merge_start).seconds/60 )



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

    # begin with uie result
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    # uie_list = read_list(filepath)
    # rerank_res = rerank_predict(args, uie_list)


    # begin with merged list
    rerank_res = rerank_predict1(args)

    # nohup python rerank_predict.py > rerank_predict.out &

