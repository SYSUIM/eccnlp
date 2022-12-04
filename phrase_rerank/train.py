#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
import numpy as np
import lambdarank
import generate_labeled_data_v2
import logging
import argparse
tf.disable_v2_behavior()
logging.basicConfig(filename="losses_v8.log", level=logging.INFO, format='%(asctime)s %(message)s')


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--DEBUG_LOG', type=bool, default=False, help='是否输出DEBUG')
parser.add_argument('--MODEL_PATH', type=str, default='./data_model_v8_lambdarank.ckpt',help='模型保存路径')
parser.add_argument('--train_data_path', type=str, default='2022-12-03_train.log',help='训练集路径')
args = parser.parse_args()

fout = open(args.train_data_path, "r")
train_data, train_data_keys = generate_labeled_data_v2.parse_labeled_data_file(fout)
fout.close()

train_data_key_count = len(train_data_keys)

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
            logging.info("-- epoch[%d] loss[%f] -- " % (
                epoch,
                loss,
            ))

        if epoch % 1000 == 0 and args.DEBUG_LOG == True:
            print ("X:\n", debug_X, mylog)
            print ("Y:\n", debug_Y, mylog)
            print ("y:\n", debug_y, mylog)
            print ("sigma_ij:\n", debug_sigma_ij, mylog)
            print ("Sij:\n", debug_Sij, mylog)
            print ("lambda_ij:\n", debug_lambda_ij, mylog)
            print ("lambda_i:\n", debug_lambda_i, mylog)
            print ("t:\n", debug_t, mylog)
            print ("tt:\n", debug_tt, mylog)
            print ("ttt:\n", debug_ttt, mylog)
    save_path = saver.save(sess, args.MODEL_PATH)
    logging.info ("Model saved in file: %s" % save_path)

