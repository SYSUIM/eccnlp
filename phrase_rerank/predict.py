#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np
import lambdarank
import generate_labeled_data_v2
import csv
import pandas as pd
import logging
import math
import argparse
from rank_data_process import form_predict_input_list, parse_labeled_data_file, get_logger
tf.disable_v2_behavior()

#此程序用于对uie抽取出原因至少有两个的样本排序，并在总数据中写入字典rerank

def add_reason(o1,reason_id,reason_list):
    o=[]
    for i in range(o1.shape[0]):
        o.append([o1[i][0],reason_list[reason_id]])
        reason_id+=1
    return reason_id, o

def add_rerank(args, rerank_list, log):
    now=0
    res=[]
    lenth=len(rerank_list)
    with open(args.path_of_merged_reasons, "r", encoding="utf8") as f:
        lines = f.readlines()
        for i in range(len(lines)):           
            data_pre = eval(lines[i])           
            data=data_pre["output"][0]
            if len(data) == 0:#uie预测无原因
                continue
            all_reason_list=[]
            if len(data[args.type]) == 0: #业绩归因为空
                all_reason_list.append(data[args.type])
            elif len(data[args.type]) < 2:#只有一个原因
                all_reason_list.append(data[args.type][0]["text"])
            else:
                for i in rerank_list[now]:
                    all_reason_list.append(i)
                now += 1
            data_pre["rerank"]=all_reason_list
            res.append(data_pre)
            log.info(data_pre)               
    return res

def predict(all_list, reasons):
    test_data, test_data_keys = parse_labeled_data_file(args, all_list)
    test_data_key_count = len(test_data_keys)
    reason_list = reasons
    reason_id=0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, args.MODEL_PATH)
        total_pairs_count = 0
        falsepositive_pairs_count = 0
        total_queries_count = test_data_key_count
        falsepositive_rank_count = 0
        rerank_list=[]
        for idx in range(test_data_key_count):
            qid = test_data_keys[idx]
            doc_list = test_data[qid]
            # rank evaluation
            X, Y = [], []
            for query_doc_vec in doc_list:
                X.append(query_doc_vec[1:])
                Y.append(query_doc_vec[0:1])
            X, Y = np.array(X), np.array(Y)
            O, o1 = sess.run([lambdarank.Y, lambdarank.y], feed_dict={lambdarank.X:X, lambdarank.Y:Y})
            reason_id, o = add_reason(o1,reason_id,reason_list)
            o_sorted=sorted(o, key=lambda x:x[0], reverse=True)
            rerank_line=[]
            for i in range(o1.shape[0]):
                rerank_line.append(o_sorted[i][1])
            rerank_list.append(rerank_line)
        return rerank_list

if __name__ == "__main__":
    logpath = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/"
    log = get_logger("predict-M15", logpath)

    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--MODEL_PATH', type=str, default='./data/data_model/model_v15_lambdarank.ckpt',help='模型保存路径')
    parser.add_argument('--type', type=str, default='业绩归因',help='原因类型')
    parser.add_argument('--path_of_merged_reasons', type=str, default='./data/res_log/2.0_2022-12-23_merge.txt',help='path of merged reasons')
    parser.add_argument('--reason_num', type=int, default=10,help='reason number')
    parser.add_argument('--f_num', type=int, default=2, help='feature number')
    args = parser.parse_args()

    all_list, reasons = form_predict_input_list(args)
    rerank_list = predict(all_list, reasons)
    res = add_rerank(args, rerank_list, log)

