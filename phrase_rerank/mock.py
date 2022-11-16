#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import config
import functools
import operator

def cmp(x,y):
    if x>y:
        return 1
    if x<y:
        return -1
    else:
        return 0

def parse_labeled_data_file(fin):
    """Read labefinled data from file
    从文件中读取标签数据
    File is assumed as organized in standard svmlight
    
    Returns:
        tuple: (map[qid]feature_vec, list[qid])
    """
    data = {}
    keys = []
    last_key = ""
    for line in fin:
        line = line.split("#")[0]
        # 1 qid:0 1:0.69 2:0.55
        elems = line.split(" ")
        label = float(elems[0])
        qid = elems[1].split(":")[1]
        feature_v = [0.0] * config.FEATURE_NUM
        # 提取line中的feature_v[1],feature_v[2]
        for i in range(2, config.FEATURE_NUM + 2):
            subelems = elems[i].split(":")
            # 1   0.69
            if len(subelems) < 2:
                continue
            index = int(subelems[0]) - 1
            feature_v[index] = float(subelems[1])
        # 
        if qid in data:
            data[qid].append([label] + feature_v)
        else:
            data[qid] = [[label] + feature_v]
        if last_key != qid:
            last_key = qid
            keys.append(qid)

    return data, keys


def calc_query_doc_pairwise_data(doc_list):
    """Calc required sample pairs from one retrival
    计算需要一次检索的样本对
    If doc_list contains A, B, C and A > B > C
         pairs are generated as A > B, A > C, B > C

    Args:
        doc_list: list of list: [score, f1, f2 , ..., fn]

    Returns:
        [X1, X2], [Y1, Y2]:
        X1.shape = X2.shape = (None, config.FEATURE_NUM)
        Y1.shape = Y2.shape = (None, 1)
    """
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    # sorted_doc_list = sorted(doc_list, cmp=lambda x, y: cmp(y[0], x[0]))
    sorted_doc_list = sorted(doc_list, key=functools.cmp_to_key(lambda x, y: cmp(y[0], x[0])))
    for i in range(len(sorted_doc_list)):
        for j in range(i + 1, len(sorted_doc_list), 1):
            X1.append(sorted_doc_list[i][1:])
            Y1.append(sorted_doc_list[i][0:1])
            X2.append(sorted_doc_list[j][1:])
            Y2.append(sorted_doc_list[j][0:1])
    return [X1, X2], [Y1, Y2]

if __name__ == "__main__":
    mylog1 = open('pointwise_data.log', mode = 'w',encoding='utf-8')
    mylog2 = open('pairtwise_data.log', mode = 'w',encoding='utf-8')
    fout = open(config.TRAIN_DATA, "r")
    data, data_keys = parse_labeled_data_file(fout)
    fout.close()
    print ("--- parsed pointwise data ---",file=mylog1)
    print (data,file=mylog1)
    print ("--- parsed pairwise data ---",file=mylog2)
    for k, v in data.items():
        print ("pairs for key [%s]:" % (k),file=mylog2)
        print (calc_query_doc_pairwise_data(v),file=mylog2)
