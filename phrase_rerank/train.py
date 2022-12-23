#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
import numpy as np
import lambdarank
import generate_labeled_data_v2
import logging
import argparse
from datetime import datetime
from rank_data_process import form_input_list, parse_labeled_data_file
tf.disable_v2_behavior()

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--DEBUG_LOG', type=bool, default=False, help='是否输出DEBUG')
parser.add_argument('--MODEL_PATH', type=str, default='./data/data_model/model_v17_lambdarank.ckpt',help='模型保存路径')
parser.add_argument('--f_num', type=int, default=2, help='feature number')
parser.add_argument('--type', type=str, default='业绩归因',help='type of answer')
parser.add_argument('--reason_num', type=int, default=10,help='reason number')
parser.add_argument('--path_of_merged_reasons', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2.0_2022-12-23_merge.txt',help='path of merged reasons')
args = parser.parse_args()

def get_logger(name, logpath):
    logger = logging.getLogger(name)
    filename = f'{datetime.now().date()}_{name}.log'
    fh = logging.FileHandler(logpath + filename, mode='a+', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    logger.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def convert_np_data(query_doc_list):
    x = []
    y = []
    for qd in query_doc_list:
        x.append(qd[1:])
        y.append(qd[:1])
    return np.array(x), np.array(y)

all_list, train_list, test_list, reason_list = form_input_list(args)
train_data, train_data_keys = parse_labeled_data_file(args, train_list)
train_data_key_count = len(train_data_keys)
logpath = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/loss/"
log=get_logger('loss_17', logpath)
saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    query_doc_index = 0
    for epoch in range(0, 5000):
        X, Y = [[], []], [[], []]
        # get next query_doc list by a query
        query_doc_index += 1
        query_doc_index %= len(train_data_keys)
        key = train_data_keys[query_doc_index]
        doc_list = train_data[key]
        # convert to graph input structure
        X, Y = convert_np_data(doc_list)
        sess.run(lambdarank.train_op, feed_dict={
            lambdarank.X:X,
            lambdarank.Y:Y,
        })
        if epoch % 100 == 0:
            loss, \
                    debug_X, debug_Y, debug_y,\
                    debug_sigma_ij, debug_Sij, debug_lambda_ij,\
                    debug_lambda_i,\
                    debug_t, debug_tt, debug_ttt = \
                    sess.run([lambdarank.loss,
                        lambdarank.X,
                        lambdarank.Y,
                        lambdarank.y,
                        lambdarank.sigma_ij,
                        lambdarank.Sij,
                        lambdarank.lambda_ij,
                        lambdarank.lambda_i,
                        lambdarank.t,
                        lambdarank.tt,
                        lambdarank.ttt
                        ],
                       feed_dict={
                           lambdarank.X:X,
                           lambdarank.Y:Y,
                       })
            log.info("-- epoch[%d] loss[%f] -- " % (
                epoch,
                loss,
            ))
    save_path = saver.save(sess, args.MODEL_PATH)
    log.info ("Model saved in file: %s" % save_path)

