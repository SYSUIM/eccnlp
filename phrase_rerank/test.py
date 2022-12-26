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
from datetime import datetime
from rank_data_process import form_input_list, parse_labeled_data_file, get_logger2
tf.disable_v2_behavior()

def convert_np_data(query_doc_list):
    x = []
    y = []
    for qd in query_doc_list:
        x.append(qd[1:])
        y.append(qd[:1])
    return np.array(x), np.array(y)

def add_reason(o1,reason_id,reason_list_test):
    o=[]
    for i in range(o1.shape[0]):
        o.append([o1[i][0],reason_list_test[reason_id]])
        reason_id+=1
    return reason_id, o

def test_model(args, test_list, logpath):
    test_data, test_data_keys = parse_labeled_data_file(args, test_list)
    test_data_key_count = len(test_data_keys)
    reason_list_test = reason_list
    reason_id = int(math.floor(len(reason_list_test)/args.reason_num*0.7)*args.reason_num)
    saver = tf.train.Saver()
    all_reason_list=[]

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, args.MODEL_PATH)    
        total_pairs_count = 0
        falsepositive_pairs_count = 0
        total_queries_count = test_data_key_count
        falsepositive_rank_count = 0
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
            reason_id, o = add_reason(o1,reason_id,reason_list_test)
            o_sorted=sorted(o, key=lambda x:x[0], reverse=True)
            all_reason_list.append(o_sorted)
            result_label = "none"

            if(args.top == "top1"):
                if (o_sorted[0][0]==o[0][0]):
                    result_label = "true_positive"
                else:
                    result_label = "falsepositive"
                    falsepositive_rank_count += 1

            if(args.top == "top2"):
                if (o_sorted[0][0]==o[0][0] and o_sorted[1][0]==o[1][0]) or (o_sorted[0][0]==o[1][0] and o_sorted[1][0]==o[0][0]):
                    result_label = "true_positive"
                else:
                    result_label = "falsepositive"
                    falsepositive_rank_count += 1

            for i in range(o1.shape[0]):
                log.info("%s\t%.2f\t%.2f\t%s\t%s" % (result_label, o[i][0], O[i], qid, o[i][1]))
        
        log.info ("-- rank  top1 precision [%d/%d = %f] -- " % (
                total_queries_count - falsepositive_rank_count,
                total_queries_count,
                1.0 - 1.0 * falsepositive_rank_count / total_queries_count
        ))
        log.info("Model restored from file: %s" % args.MODEL_PATH)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--MODEL_PATH', type=str, default='./data/data_model/model_v16_lambdarank.ckpt',help='模型保存路径')
    parser.add_argument('--f_num', type=int, default=2, help='feature number')
    parser.add_argument('--type', type=str, default='业绩归因',help='type of answer')
    parser.add_argument('--reason_num', type=int, default=10,help='reason number')
    parser.add_argument('--path_of_merged_reasons', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2.0_2022-12-23_merge.txt',help='path of merged reasons')
    # parser.add_argument('--top', type=str, default='top1', help='choose top1 or top2')
    parser.add_argument('--top', type=str, required=True, help='choose top1 or top2')
    args = parser.parse_args()

    logpath = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/"
    if(args.top == "top1"):
        log = get_logger2('top1-M16',logpath)
    if(args.top == "top2"):
        log = get_logger2('top2-M16',logpath)
    
    all_list, train_list, test_list, reason_list = form_input_list(args)
    test_model(args, test_list, logpath)


