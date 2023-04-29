# phrase rerank
import sys
sys.path.append('/data/fkj2023/Project/eccnlp_1')
from rank_data_process import form_input_list, add_embedding, get_text_list, merge_reasons, read_word, uie_list_filter, split_rerank_data, form_predict_input_list
from lambdarank import LambdaRank, train_rerank, validate_rerank, precision_k, predict_rank, add_rerank
from datetime import datetime
import numpy as np
import torch
from utils import read_list_file, check_log_dir
# import config
from data_process.dataprocess import build_thesaurus
import logging
import argparse


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
    # training_data = training_data[:,:4]
    # training_data = np.concatenate([training_data[:,:2], training_data[:,4:-1]], axis=1)
    model = LambdaRank(training_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_rerank(args, training_data, device, model)


    # evaluate
    test_list, test_reason = form_predict_input_list(args, test_merged_list, word)
    test_data = np.array(test_list)

    ndcg , pred_scores= validate_rerank(args, test_data, 2)
    precision_k(args, test_data)

    predicted_list_rerank, rerank_reasons_test, rerank_scores_test = predict_rank(args, test_data, test_reason)
    rerank_on_test_data = add_rerank(args, rerank_reasons_test, rerank_scores_test, test_merged_list)
   

    return rerank_on_test_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='run rerank')
    parser.add_argument('--type', type=str, default='业绩归因',help='type of answer')
    parser.add_argument('--f_num', type=int, default=34, help='feature number')
    parser.add_argument('--vocab_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/bert_model/vocab.txt',help='vocab path')
    parser.add_argument('--code_length', type=int, default=16,help='the dimension of sentence features') 
    parser.add_argument('--t_path', type=str, default='/data/fkj2023/Project/eccnlp_local/data_process/stop_words.txt',help='path of stop words')
    parser.add_argument("--rerank_save_path", default='./checkpoint/rerank_model/' + datetime.now().strftime("%m_%d_%H_%M_%S")+'.pkl', type=str, help="path that rerank model will be save.")
    parser.add_argument('--rerank_learning_rate', default=0.0001, type=float, help='learning rate of rerank')
    parser.add_argument('--rerank_epoch', default=3, type=int, help='Total number of training epochs of rerank')
    args = parser.parse_args()

    word = read_word('/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/2022-12-24_word.log')


    # train rerank 
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/test.txt'
    filepath = '/data/fkj2023/Project/eccnlp_1/log/20230427v3/uieres_result.txt'
    # uie 结果路径
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/info.txt'
    # filepath = ''

    uie_list = read_list_file(filepath)  
    rerank_on_test_data = run_rerank(args, uie_list, word, [])    

    with open("./rerank_res_on_testdata"+".txt", 'w') as f:
        [f.write(str(data) + '\n') for data in rerank_on_test_data]



# nohup python train_rerank.py > run_rerank.log &