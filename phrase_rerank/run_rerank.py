import argparse
# import ..config
# from data_process.dataprocess import re_pattern2, train_dataset, build_thesaurus
from rank_data_process import get_logger, get_logger2, form_input_list, read_list, print_list, add_embedding, get_text_list, merge_reasons, read_word
import numpy as np
from lambdarank import LambdaRank, train, validate, precision_k
import torch
from datetime import datetime
# from ..data_process.dataprocess import build_thesaurus


# def update_thesaurus(args):
#     raw_dataset = read_list_file(args.data)
#     dataset = re_filter(raw_dataset)
#     t_path = '/data/fkj2023/Project/eccnlp_local/data_process/stop_words.txt'
#     word = build_thesaurus(dataset, t_path)
#     logpath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/word/'
#     log = get_logger('word',logpath)
#     print_list(word, log)
#     return word


#rerank
def run_rerank(args, uie_list, word):
    
    #embedding
    embedding_start = datetime.now()
    after_embedding_list = add_embedding(args, uie_list)
    logpath1 = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/embedding/'
    log1 = get_logger("embedding_list",logpath1)
    print_list(after_embedding_list, log1)
    embedding_end = datetime.now()
    log1.info("embedding time : %s  minutes", (embedding_end - embedding_start).seconds/60 )

    #merge reasons
    merge_start = datetime.now()
    text_list, num_list = get_text_list(uie_list)
    merged_list = merge_reasons(args, text_list, num_list, after_embedding_list)
    logpath2 = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/merged/'
    log2 = get_logger("merged_list",logpath2)
    print_list(merged_list, log2)
    merge_end = datetime.now()
    log2.info("merge time : %s  minutes", (merge_end - merge_start).seconds/60 )

    #train
    train_start = datetime.now()
    logpath3 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/train_lambdarank/" 
    # log3 = get_logger2('test_pythorch_main_res',logpath3)
    log3 = get_logger2('train_ndcg',logpath3)
    epoch = 10
    learning_rate = 0.0001
    all_list, train_list, test_list, reason_of_test = form_input_list(args, merged_list, word)
    training_data = np.array(train_list)
    model = LambdaRank(training_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    modelpath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_v0_parameter'+datetime.now().strftime("%m_%d_%H_%M_%S")+'.pkl'
    train(training_data, learning_rate, epoch, modelpath, device, model, log3)
    train_end = datetime.now()
    log3.info("train time : %s  minutes", (train_end - train_start).seconds/60 )

    # value
    value_start = datetime.now()
    validate_data = np.array(test_list)
    k = 2
    ndcg , pred_scores= validate(validate_data, k, modelpath)
    # log3.info("pred_scores: %s", pred_scores)
    log3.info("np.nanmean(ndcg): %s", np.nanmean(ndcg))
    precision_k(validate_data, modelpath, log3)
    value_end = datetime.now()
    log3.info("value time : %s  minutes", (value_end - value_start).seconds/60 )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='run rerank')
    parser.add_argument('--type', type=str, default='业绩归因',help='type of answer')
    parser.add_argument('--f_num', type=int, default=34, help='feature number')
    parser.add_argument('--vocab_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/bert_model/vocab.txt',help='vocab path')
    parser.add_argument('--code_length', type=int, default=16,help='the dimension of sentence features') 
    args = parser.parse_args()

    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/test.txt'
    filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    # uie 结果路径
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/info.txt'

    uie_list = read_list(filepath)
    # word = update_thesaurus(args)
    word = read_word('/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/2022-12-24_word.log')
    run_rerank(args, uie_list, word)    
