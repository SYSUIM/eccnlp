import numpy as np
import csv
import pandas as pd
import logging
import math
import argparse
from datetime import datetime
from bert import sentence_features

parser = argparse.ArgumentParser(description='embedding')
parser.add_argument('--type', type=str, default='业绩归因',help='原因类型')
args = parser.parse_args()

def get_logger(name):
    logger = logging.getLogger(name)
    filename = f'{datetime.now().date()}_{name}.txt'
    fh = logging.FileHandler("./data/res_log/2.0_"+ filename, mode='w+', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

log=get_logger('add_embedding')

def add_embedding(args, filepath):
    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()       
        for i in range(len(lines)):
            # dic = {}
            if lines[i][0] != '{':
                continue
            data_pre = eval(lines[i])
            dic = data_pre
            if len(data_pre["output"][0]) == 0:  # 预测为无因果
                log.info(dic)
                continue
            data=data_pre["output"][0]
            #data: {'业绩归因': [{'text': '“调整年”', 'start': 3, 'end': 8, 'probability': 0.6058174246136119}]}
            elem_num=len(data[args.type])                 
            if elem_num>0:  #至少预测出一个归因  
                dic_new = {}  
                uie_re = []
                str_pre = data_pre["content"]
                s_cls = '[CLS]'
                s_sep = '[SEP]'
                #对每个归因，构造上下文字符串
                for j in data[args.type]:           
                    s_before = s_cls + str_pre[0 : j["start"]] + s_sep
                    s_after = s_cls + str_pre[j["end"] : ] + s_sep
                    sen_f = sentence_features([s_before, s_after])
                    j['s_before'] = sen_f[0]
                    j['s_after'] = sen_f[1]
                    uie_re.append(j)
                dic_new[args.type] = uie_re
            dic['output'] = [dic_new]
            log.info(dic)
               
                    
if __name__ == '__main__':
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/test.txt'
    filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    add_embedding(args, filepath)
