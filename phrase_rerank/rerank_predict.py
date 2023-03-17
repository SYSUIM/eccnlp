import argparse
from rank_data_process import get_logger1,  form_predict_input_list, read_list_file, print_list, add_embedding, get_text_list, merge_reasons,add_embedding_new,  uie_list_filter, read_word
import numpy as np
from lambdarank import LambdaRank, add_rerank, predict_rank, precision_k
import torch
from datetime import datetime


def rerank_predict1(args):

    predict_start = datetime.now()
    word = read_word(args.word_path)

    # begin with merged list
    filepath ='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/merged/2023-03-09_13_23_57_merged_list.log'
    merged_list = read_list_file(filepath)

    #predict   
    predict_list, reasons = form_predict_input_list(args, merged_list, word)
    predict_data = np.array(predict_list)
    predicted_list, rerank_reasons, rerank_scores = predict_rank(args, predict_data, reasons)
    logpath4 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/predict/" 
    log4 = get_logger1('result_add_rerank',logpath4)
    res = add_rerank(args, rerank_reasons,rerank_scores, merged_list, log4)
    predict_end = datetime.now()
    log4.info("predict time : %s  minutes", (predict_end - predict_start).seconds/60)
    precision_k(predict_data, args.lambdarank_path, log4)
    log4.info("lambdarank model path: %s", args.lambdarank_path)
    return res 

def rerank_predict(args, uie_list):

    logpath9 = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/embedding/'
    log9 = get_logger1("record_predict_time_cost",logpath9)

    # begin with uie result
    # predict_start = datetime.now()
    word = read_word(args.word_path)

    #embedding
    embedding_start = datetime.now()
    # after_embedding_list = add_embedding(args, uie_list)

    filtered_uie_list, context_list = uie_list_filter(args, uie_list)
    print("filter end")
    after_embedding_list = add_embedding(args, filtered_uie_list)
    
    logpath1 = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/embedding/'
    log1 = get_logger1("pre_embedding_list",logpath1)
    print_list(after_embedding_list, log1)
    # exit(0)
    embedding_end = datetime.now()
    log9.info("embedding time : %s  minutes", (embedding_end - embedding_start).seconds/60 )  

    #merge reasons
    merge_start = datetime.now()
    text_list, num_list = get_text_list(uie_list)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)
    logpath2 = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/merged/'
    log2 = get_logger1("pre_merged_list",logpath2)
    print_list(merged_list, log2)
    merge_end = datetime.now()
    log9.info("merge time : %s  minutes", (merge_end - merge_start).seconds/60 )



    #predict   
    predict_start = datetime.now()
    predict_list, reasons = form_predict_input_list(args, merged_list, word)
    predict_data = np.array(predict_list)
    predicted_list, rerank_reasons, rerank_scores = predict_rank(args, predict_data, reasons)
    logpath4 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/predict/" 
    log4 = get_logger1('result_add_rerank',logpath4)
    res = add_rerank(args, rerank_reasons,rerank_scores, merged_list, log4)
    predict_end = datetime.now()
    print("predict time (minutes):")
    print((predict_end - predict_start).seconds/60)
    log9.info("predict time : %s  minutes", (predict_end - predict_start).seconds/60)
    # precision_k(predict_data, args.lambdarank_path, log4)
    # log4.info("lambdarank model path: %s", args.lambdarank_path)
    return res 

def rerank_predict2(args, uie_list):

    # predict_start = datetime.now()
    word = read_word(args.word_path)

    # begin with embedding list
    filepath ='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/embedding/2023-03-06_00_14_55_pre_embedding_list.log'

    after_embedding_list = read_list_file(filepath)
    text_list, num_list = get_text_list(uie_list)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)    
    logpath2 = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/merged/'
    log2 = get_logger1("pre_merged_list",logpath2)
    print_list(merged_list, log2)

    #predict   
    predict_list, reasons = form_predict_input_list(args, merged_list, word)
    predict_data = np.array(predict_list)
    predicted_list, rerank_reasons, rerank_scores = predict_rank(args, predict_data, reasons)
    logpath4 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/predict/" 
    log4 = get_logger1('result_add_rerank_M0',logpath4)
    res = add_rerank(args, rerank_reasons,rerank_scores, merged_list, log4)
    # predict_end = datetime.now()
    # log4.info("predict time : %s  minutes", (predict_end - predict_start).seconds/60)
    # precision_k(predict_data, args.lambdarank_path, log4)
    # log4.info("lambdarank model path: %s", args.lambdarank_path)
    return res 


def filter_res(res):
    filtered_res = []
    for i in range(len(res)):
        data_pre = res[i]
        if len(data_pre["score"]) != 0:
            filtered_res.append(data_pre)  

    return filtered_res

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='rerank_predict')
    parser.add_argument('--type', type=str, default='业绩归因',help='type of answer')
    parser.add_argument('--f_num', type=int, default=34, help='feature number')
    parser.add_argument('--vocab_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/bert_model/vocab.txt',help='vocab path')
    parser.add_argument('--code_length', type=int, default=16,help='the dimension of sentence features') 
    parser.add_argument('--lambdarank_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_v0_parameter_bak.pkl',help='lambdarank path')
    parser.add_argument('--word_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/2022-12-24_word.log',help='word path')
    args = parser.parse_args()


    # filepath = '/data/fkj2023/Project/eccnlp_local/data/2023-03-14_3.1_nocut_inference.log'
    # log1 =get_logger1('2023-03-14_3.1_nocut_inference_filtered','/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/inference/')
    # # filepath = ''
    # res = read_list_file(filepath)
    # filtered_res = filter_res(res)
    # print_list(filtered_res, log1)

    # exit()

    # begin with uie result
    filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    # filepath = '/data/fkj2023/Project/eccnlp_local/data/info_extraction_result_1222_test.txt'
    # filepath = '/data/pzy2022/project/eccnlp/info_extraction/after_extraction_data3.1.txt'
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/inferenceDoubleEnsemble_prob0.9.log'
    # filepath ='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/compare/inferenceDoubleEnsemble_prob0.9.log'
    uie_list = read_list_file(filepath)
    rerank_res = rerank_predict(args, uie_list)


    # begin with merged list
    # rerank_res = rerank_predict1(args)

    # nohup python rerank_predict.py > rerank_predict.out &

