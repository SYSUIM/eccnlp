import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from datetime import datetime
import logging
# from rank_data_process import get_logger, get_logger2, form_input_list, form_predict_input_list, read_list,add_embedding, get_text_list, merge_reasons


def dcg(scores):
    """
    compute the DCG value based on the given score
    :param scores: a score list of documents
    :return v: DCG value
    """
    v = 0
    for i in range(len(scores)):
        v += (np.power(2, scores[i]) - 1) / np.log2(i+2)  # i+2 is because i starts from 0
    return v


def idcg(scores):
    """
    compute the IDCG value (best dcg value) based on the given score
    :param scores: a score list of documents
    :return:  IDCG value
    """
    best_scores = sorted(scores)[::-1]
    return dcg(best_scores)


def ndcg(scores):
    """
    compute the NDCG value based on the given score
    :param scores: a score list of documents
    :return:  NDCG value
    """
    return dcg(scores)/idcg(scores)

def single_dcg(scores, i, j):
    """
    compute the single dcg that i-th element located j-th position
    :param scores:
    :param i:
    :param j:
    :return:
    """
    return (np.power(2, scores[i]) - 1) / np.log2(j+2)

def map():


    return 


def ndcg_k(scores, k):
    scores_k = scores[:k]
    dcg_k = dcg(scores_k)
    idcg_k = dcg(sorted(scores)[::-1][:k])
    if idcg_k == 0:
        return np.nan
    return dcg_k/idcg_k


def group_by(data, qid_index):
    """
    :param data: input_data
    :param qid_index: the column num where qid locates in input data
    :return: a dict group by qid
    """
    qid_doc_map = {}
    idx = 0
    for record in data:
        qid_doc_map.setdefault(record[qid_index], [])
        qid_doc_map[record[qid_index]].append(idx)
        idx += 1
    return qid_doc_map


def get_pairs(scores):
    """

    :param scores: given score list of documents for a particular query
    :return: the documents pairs whose firth doc has a higher value than second one.
    """
    pairs = []
    for i in range(len(scores)):
        for j in range(len(scores)):
            if scores[i] > scores[j]:
                pairs.append((i, j))
    return pairs


def compute_lambda(true_scores, temp_scores, order_pairs, qid):
    """

    :param true_scores: the score list of the documents for the qid query
    :param temp_scores: the predict score list of the these documents
    :param order_pairs: the partial oder pairs where first document has higher score than the second one
    :param qid: specific query id
    :return:

        lambdas: changed lambda value for these documents
        w: w value
        qid: query id
    """
    doc_num = len(true_scores)
    lambdas = np.zeros(doc_num)
    w = np.zeros(doc_num)
    IDCG = idcg(true_scores)
    single_dcgs ={}
    for i, j in order_pairs:
        if (i, i) not in single_dcgs:
            single_dcgs[(i, i)] = single_dcg(true_scores, i, i)
        if (j, j) not in single_dcgs:
            single_dcgs[(j, j)] = single_dcg(true_scores, j, j)
        single_dcgs[(i, j)] = single_dcg(true_scores, i, j)
        single_dcgs[(j, i)] = single_dcg(true_scores, j, i)


    for i, j in order_pairs:
        delta = abs(single_dcgs[(i,j)] + single_dcgs[(j,i)] - single_dcgs[(i,i)] -single_dcgs[(j,j)])/IDCG
        rho = 1 / (1 + np.exp(temp_scores[i] - temp_scores[j]))  
        lambdas[i] += rho * delta
        lambdas[j] -= rho * delta

        rho_complement = 1.0 - rho   
        w[i] += rho * rho_complement * delta   
        w[j] -= rho * rho_complement * delta   

    return lambdas, w, qid


class LambdaRank(nn.Module):      
    def __init__(self, training_data):
        super(LambdaRank, self).__init__()
     
        self.training_data = training_data
        self.n_feature = training_data.shape[1] - 2
        self.h1_units = 512
        self.h2_units = 256
        self.trees = [] 

        self.h1 = nn.Linear(training_data.shape[1] - 2, 512)
        self.h2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
        # for para in self.model.parameters():
        #     print(para[0])

    def forward(self, x ):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.out(x)
        return x

def train_rerank(args, training_data, device, model):
    """
    train the model to fit the train dataset
    """

    qid_doc_map = group_by(training_data, 1)
    query_idx = qid_doc_map.keys()
    # true_scores is a matrix, different rows represent different queries
    true_scores = [training_data[qid_doc_map[qid], 0] for qid in query_idx]

    order_paris = []
    for scores in true_scores:
        order_paris.append(get_pairs(scores))

    sample_num = len(training_data)

    for i in range(args.rerank_epoch):
        train_data = torch.from_numpy(training_data[:, 2:].astype(np.float32)).to(device)
        predicted_scores = model(train_data)
        predicted_scores_numpy = predicted_scores.cpu().data.numpy()
        lambdas = np.zeros(sample_num)

        pred_score = [predicted_scores_numpy[qid_doc_map[qid]] for qid in query_idx]

        zip_parameters = zip(true_scores, pred_score, order_paris, query_idx)
        for ts, ps, op, qi in zip_parameters:
            sub_lambda, sub_w, qid = compute_lambda(ts, ps, op, qi)
            lambdas[qid_doc_map[qid]] = sub_lambda

        # update parameters
        model.zero_grad()
        lambdas_torch = torch.Tensor(lambdas).view((len(lambdas), 1)).to(device)
        predicted_scores.backward(lambdas_torch, retain_graph=True)  
        with torch.no_grad():
            for param in model.parameters():
                param.data.add_(param.grad.data * args.rerank_learning_rate)


        if i % 1 == 0:
            qid_doc_map = group_by(training_data, 1)
            ndcg_list = []
            for qid in qid_doc_map.keys():
                subset = qid_doc_map[qid]
                X_subset = torch.from_numpy(training_data[subset, 2:].astype(np.float32)).to(device)
                sub_pred_score = model(X_subset).cpu().data.numpy().reshape(1, len(X_subset)).squeeze()

                # calculate the predicted NDCG
                true_label = training_data[qid_doc_map[qid], 0]
                k = len(true_label)
                pred_sort_index = np.argsort(sub_pred_score)[::-1]
                true_label = true_label[pred_sort_index]
                ndcg_val = ndcg_k(true_label, k)
                ndcg_list.append(ndcg_val)
            logging.info('Epoch:{}, Average NDCG : {}'.format(i, np.nanmean(ndcg_list)))
    logging.info(model.state_dict().keys())   # output model parameter name
    torch.save(model.state_dict(), args.rerank_save_path)
    logging.info("model saved in %s", args.rerank_save_path)



def predict_rank(args, data, reason_list):

    model = LambdaRank(data)
    model.load_state_dict(torch.load(args.lambdarank_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    qid_doc_map = group_by(data, 1)
    predicted_scores = np.zeros(len(data))
    predicted_list = []

    rerank_scores = []
    rerank_reasons= []
    for qid in qid_doc_map.keys():    
        predicted_scores_after_rerank = []    
        subset = qid_doc_map[qid] 
        X_subset = torch.from_numpy(data[subset, 2:].astype(np.float32)).to(device)
        sub_pred_score = model(X_subset).cpu().data.numpy().reshape(1, len(X_subset)).squeeze()
        if sub_pred_score.size == 1:          
            sub_pred_score = [float(sub_pred_score)]
        else: 
            sub_pred_score = sub_pred_score.tolist()

        # rerank reasons by scores
        predicted_scores[qid_doc_map[qid]] = sub_pred_score 
        rerank_qid_reasons = []  
        pred_sort_index = np.argsort(sub_pred_score)[::-1]
        for i in pred_sort_index:
            predict_score = sub_pred_score[i] 
            predicted_scores_after_rerank.append(sub_pred_score[i])       
            subset_idex = sub_pred_score.index(predict_score)
            data_index = subset[subset_idex]
            rerank_qid_reasons.append(reason_list[data_index])
        rerank_scores.append(predicted_scores_after_rerank)
        rerank_reasons.append(rerank_qid_reasons)

        for i in sub_pred_score:
            predicted_list.append(i)
    return predicted_list, rerank_reasons, rerank_scores

def validate_rerank(args,data, k):
    """
    validate the NDCG metric
    :param data: given the testset
    :param k: used to compute the NDCG@k
    :return:
    """
    model = LambdaRank(data)
    model.load_state_dict(torch.load(args.rerank_save_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    qid_doc_map = group_by(data, 1)
    ndcg_list = []
    predicted_scores = []
    for qid in qid_doc_map.keys():
        subset = qid_doc_map[qid]
        X_subset = torch.from_numpy(data[subset, 2:].astype(np.float32)).to(device)
        sub_pred_score = model(X_subset).cpu().data.numpy().reshape(1, len(X_subset)).squeeze()        
        # calculate the predicted NDCG
        true_label = data[qid_doc_map[qid], 0]
        predicted_scores.append([sub_pred_score, true_label])
        k = len(true_label)
        pred_sort_index = np.argsort(sub_pred_score)[::-1]
        true_label = true_label[pred_sort_index]
        ndcg_val = ndcg_k(true_label, k)
        ndcg_list.append(ndcg_val)
    logging.info("np.nanmean(ndcg): %s", np.nanmean(ndcg_list))
    return ndcg_list, predicted_scores

def precision_k(args, data):

    model = LambdaRank(data)
    model.load_state_dict(torch.load(args.rerank_save_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    qid_doc_map = group_by(data, 1)
    true_predict = 0
    predicted_scores = []
    for qid in qid_doc_map.keys():
        # log.info('------------------a new qid-------------------: %s',qid)
        subset = qid_doc_map[qid]  # index
        real_score = data[subset, 0] # real_score
        # log.info("real_score : %s", real_score)
        X_subset = torch.from_numpy(data[subset, 2:].astype(np.float32)).to(device)  # 34-dimensional features
        sub_pred_score = model(X_subset).cpu().data.numpy().reshape(1, len(X_subset)).squeeze()
        # log.info("sub_pred_score: %s", sub_pred_score)  
        rank1_index = np.argmax(sub_pred_score)
        if(rank1_index == 0):
            true_predict += 1
        # log.info("rank1_index: %s", rank1_index)
        predicted_scores.append([sub_pred_score, real_score])
        pred_sort_index = np.argsort(sub_pred_score)[::-1]
        # log.info("pred_sort_index:%s", pred_sort_index)
    precision_1 = true_predict / len(qid_doc_map)
    logging.info("rerank precision_1: %s",precision_1)

    # log.info("qid_doc_map.keys(): %s", qid_doc_map.keys())
    return precision_1, predicted_scores



def add_rerank(args, rerank_list,rerank_scores, merged_list):
    now=0
    res=[]
    lines = merged_list
    for i in range(len(lines)):           
        data_pre = lines[i]     
        data=data_pre["output"][0]
        if len(data) == 0:
            continue
        all_reason_list=[]
        if len(data[args.type]) == 0: 
            all_reason_list.append(data[args.type])
            data_pre["rerank"]=all_reason_list
            data_pre["score"]=[]
        else:
            data_pre["rerank"] = rerank_list[now]
            data_pre["score"] = rerank_scores[now]
            now += 1
        res.append(data_pre)
        # log.info(data_pre)               
    return res

def read_word(filepath):
    alist = []
    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            alist.append(line)
    return alist

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--type', type=str, default='业绩归因',help='type of answer')
    parser.add_argument('--reason_num', type=int, default=10,help='reason number')
    parser.add_argument('--f_num', type=int, default=34, help='feature number')
    parser.add_argument('--usage', type=str, default="train", help='generate train data or predict data')  
    parser.add_argument('--vocab_path', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/bert_model/vocab.txt',help='vocab path')
    parser.add_argument('--code_length', type=int, default=16,help='the dimension of sentence features') 
    args = parser.parse_args()

    # # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/test.txt'
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    # # uie 结果路径
    # # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/info.txt'

    # uie_list = read_list(filepath)

    # #embedding
    # after_embedding_list = add_embedding(args, uie_list)
    # #merge reasons
    # rea_list, num_list = get_reason_list(uie_list)
    # merged_list = merge_reasons_new(args, rea_list, num_list, after_embedding_list)


    # logpath2 = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/transfer/" 
    # log2 = get_logger2('gpu_train_pythorch',logpath2)

    # #train
    # epoch = 1000
    # learning_rate = 0.0001 
    # all_list, train_list, test_list, reason_of_test = form_input_list(args, merged_list)
    # training_data = np.array(train_list)
    # model = LambdaRank(training_data)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # modelpath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_v0_parameter'+datetime.now().strftime("%m_%d_%H_%M_%S")+'.pkl'
    # train(training_data, learning_rate, epoch, modelpath, device, model, log2)


    word = read_word('/data/fkj2023/Project/eccnlp_local/data/word.log')
    filepath ='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/merged/2023-03-02_merged_list.log'
    merged_list = read_list(filepath)
    modelpath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_v0_parameter.pkl'
    predict_list, reasons = form_predict_input_list(args, merged_list, word)
    predict_data = np.array(predict_list)
    predicted_list, rerank_reasons, rerank_scores = predict(predict_list, reasons ,modelpath)













    # logpath = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/transfer/" 
    # log = get_logger2('gpu_train_pythorch',logpath)
    # log2 = get_logger1('gpu_test_pythorch_res',logpath)
    # log.info('---------------------------------------------------TEST AGAIN---------------------------------------------------')
    # # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/merged_23-0115_test.txt'
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2.0_2023-01-15_merge.txt'
    # merged_list = read_list(filepath)
    # # all_list, reasons = form_predict_input_list(args, merged_list)
    # all_list, train_list, test_list, reason_of_test = form_input_list(args, merged_list)


    # training_data = np.array(train_list)
    # n_feature = training_data.shape[1] - 2
    # h1_units = 512
    # h2_units = 256
    # epoch = 40
    # learning_rate = 0.0001
    # log.info("lambda2_pytorch")



    # #train
    # modelpath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_v1_parameter.pkl'
    # model = LambdaRank(training_data)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # model.to(device)
    # print("model to device")
    # train(learning_rate,modelpath,device)
    # log.info("lambda2_pytorch train and save") 



    # # predict and validate
    # # model = LambdaRank(training_data, n_feature, h1_units, h2_units)
    # model.load_state_dict(torch.load(modelpath))
    # log.info("load model in %s",modelpath)
    # k = 2
    # test_data = np.array(test_list)
    # ndcg , pred_scores= validate(test_data, k)
    # precision_k(test_data)
    # predicted_list, rerank_reasons, rerank_scores = predict(test_data, reason_of_test,model)
    # # add_rerank(args, rerank_reasons,rerank_scores, merged_list, log2)
    # log.info("model used")
    # log.info("----------------ndcg---------------------")
    # log.info(ndcg)
    # log.info('----------------------predict_scores-----------------')
    # log.info(pred_scores)
    # log.info("----------------ndcg.shape---------------------")
    # log.info(np.array(ndcg).shape)
    # log.info("----------------Average NDCG---------------------")
    # log.info(np.nanmean(ndcg))





'''

    # predict and precision
    model = LambdaRank(training_data, n_feature, h1_units, h2_units)
    model.load_state_dict(torch.load("/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_parameter.pkl"))
    log.info("load model in /data/fkj2023/Project/eccnlp_local/phrase_rerank/data/data_model/model_parameter.pkl")
    train_data = np.array(all_list)
    precision_k(train_data)
    ndcg , pred_scores= validate(train_data, 2)
    log.info("pred_scores: %s", pred_scores)
    # log.info("np.nanmean(ndcg): %s", np.nanmean(ndcg))
    
    log.info("reason_of_test: %s", reasons)
    predicted_list, rerank_reasons, rerank_scores = predict(train_data, reasons,model)
    add_rerank(args, rerank_reasons,rerank_scores, merged_list, log2)
'''

