#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
# import tensorflow as tf
import numpy as np
import lambdarank
import mock
import config
import logging
tf.disable_v2_behavior()
logging.basicConfig(filename="top1_v7.log", level=logging.INFO, format='%(asctime)s %(message)s')


fout = open(config.TEST_DATA, "r")
test_data, test_data_keys = mock.parse_labeled_data_file(fout)
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

saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, config.MODEL_PATH)
    print("Model restored from file: %s" % config.MODEL_PATH)
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
        print("X:",X)
        print("Y:",Y)
        print(lambdarank.Y, lambdarank.y)
        O, o = sess.run([lambdarank.Y, lambdarank.y], feed_dict={lambdarank.X:X, lambdarank.Y:Y})
        # O, o = sess.run([lambdarank.Y], feed_dict={lambdarank.X:X})
        print("O:",O)
        print("o:",o)
        true_label_index = 0
        positive_label_index = 0
        max_true_value = -10000.0
        max_positive_value = -10000.0
        for i in range(o.shape[0]):
            if O[i] > max_true_value:
                max_true_value = O[i]
                true_label_index = i
            if o[i] > max_positive_value:
                max_positive_value = o[i]
                positive_label_index = i
        result_label = "true_positive"
        if true_label_index != positive_label_index:
            result_label = "falsepositive"
            falsepositive_rank_count += 1
        for i in range(o.shape[0]):
            logging.info("%s\t%.2f\t%.2f\t%s" % (result_label, o[i], O[i], qid))

    logging.info ("-- rank top1 precision [%d/%d = %f] -- " % (
            total_queries_count - falsepositive_rank_count,
            total_queries_count,
            1.0 - 1.0 * falsepositive_rank_count / total_queries_count
    ))
