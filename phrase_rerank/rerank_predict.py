import argparse

from rank_data_process import get_logger1,  form_predict_input_list, read_list, print_list, add_embedding, get_text_list, merge_reasons, read_word
import numpy as np
from lambdarank import LambdaRank, add_rerank, predict
import torch
from datetime import datetime



def rerank_predict(args, uie_list):

    predict_start = datetime.now()
    word = read_word(args.word_path)
    after_embedding_list = add_embedding(args, uie_list)
    text_list, num_list = get_text_list(uie_list)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)

    #predict   
    predict_list, reasons = form_predict_input_list(args, merged_list, word)
    predict_data = np.array(predict_list)
    # modelpath1  = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_v0_parameter.pkl'
    predicted_list, rerank_reasons, rerank_scores = predict(args, predict_data, reasons)
    logpath4 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/predict/" 
    log4 = get_logger1('result_add_rerank',logpath4)
    res = add_rerank(args, rerank_reasons,rerank_scores, merged_list, log4)
    predict_end = datetime.now()
    # log4.info("predict time : %s  minutes", (predict_end - predict_start).seconds/60)
    return res 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='rerank_predict')
    parser.add_argument('--type', type=str, default='业绩归因',help='type of answer')
    parser.add_argument('--f_num', type=int, default=34, help='feature number')
    parser.add_argument('--vocab_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/bert_model/vocab.txt',help='vocab path')
    parser.add_argument('--code_length', type=int, default=16,help='the dimension of sentence features') 
    parser.add_argument('--lambdarank_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_v0_parameter.pkl',help='lambdarank path')
    parser.add_argument('--word_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/2022-12-24_word.log',help='word path')
    args = parser.parse_args()


    # merged_list = read_list('/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/merged/2023-03-02_merged_list.log')
    # word = read_word("/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/2022-12-24_word.log")

    filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    uie_list = read_list(filepath)
    rerank_res = rerank_predict(args, uie_list)

