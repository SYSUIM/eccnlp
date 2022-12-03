#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
# import tensorflow as tf
import numpy as np
import lambdarank
import generate_labeled_data_v2
import config
import csv
import pandas as pd
import logging
import argparse
tf.disable_v2_behavior()

logging.basicConfig(filename="top2_v8.log", level=logging.INFO, format='%(asctime)s %(message)s')
parser = argparse.ArgumentParser(description='predict')
parser.add_argument('--MODEL_PATH', type=str, default='./data_model_v8_lambdarank.ckpt',help='模型保存路径')
parser.add_argument('--test_data_path', type=str, default='2022-12-03_test.log',help='测试集路径')
parser.add_argument('--result_path', type=str, default='result1203.csv',help='结果文件路径')
parser.add_argument('--reason_path', type=str, default='2022-12-03_reason_of_test.log',help='测试集原因路径')

args = parser.parse_args()


def get_reason_list_test():
    reason_list_test=[]
    with open(args.reason_path, "r", encoding="utf8") as f1:
        reasons = f1.readlines()
        for i in reasons:
            reason_list_test.append(i.strip('\n'))
    f1.close()
    return reason_list_test

fout = open(args.test_data_path, "r")
test_data, test_data_keys = generate_labeled_data_v2.parse_labeled_data_file(fout)

fout.close()

test_data_key_count = len(test_data_keys)

def convert_np_data(query_doc_list):
    """Convert query doc list to numpy data of one retrival

    Args:
        query_doc_list: list of list: [score, f1, f2 , ..., fn]

    Return:
        X, Y: [feature_vec], [label]
    """
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

reason_id=int(test_data_key_count/3*7*5)
saver = tf.train.Saver()
all_reason_list=[]
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, args.MODEL_PATH)
    print("Model restored from file: %s" % args.MODEL_PATH)
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
        reason_list_test = get_reason_list_test()
        reason_id, o = add_reason(o1,reason_id,reason_list_test)
        o_sorted=sorted(o, key=lambda x:x[0], reverse=True)
        all_reason_list.append(o_sorted)
        result_label = "none"
        if (o_sorted[0][0]==o[0][0] and o_sorted[1][0]==o[1][0]) or (o_sorted[0][0]==o[1][0] and o_sorted[1][0]==o[0][0]):
            result_label = "true_positive"
        else:
            result_label = "falsepositive"
            falsepositive_rank_count += 1
        for i in range(o1.shape[0]):
            logging.info("%s\t%.2f\t%.2f\t%s\t%s" % (result_label, o[i][0], O[i], qid, o[i][1]))
       
    logging.info ("-- rank  top2 precision [%d/%d = %f] -- " % (
            total_queries_count - falsepositive_rank_count,
            total_queries_count,
            1.0 - 1.0 * falsepositive_rank_count / total_queries_count
    ))
# 将排序后的文本写入result.csv
b=np.array(all_reason_list)
b=np.reshape(b,(-1,10)) 
data=pd.read_csv('all_content.csv')
list1=data.values.tolist()
a=np.array(list1)
c=np.hstack((a,b))
fd = open(args.result_path,'w',encoding="utf-8-sig")
writer = csv.writer(fd)
header=['content','label','top1_score','top1_reason','top2_score','top2_reason','top3_score','top3_reason','top4_score','top4_reason','top5_score','top5_reason']
writer.writerow(header)
for i in range(c.shape[0]):
    writer.writerow(c[i])
fd.close()
