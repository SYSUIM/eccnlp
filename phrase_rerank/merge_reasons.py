import numpy as np
import csv
import pandas as pd
import logging
import math
import argparse
from datetime import datetime

def get_logger(name, logpath):
    logger = logging.getLogger(name)
    filename = f'{datetime.now().date()}_{name}.txt'
    fh = logging.FileHandler(logpath + "2.0_"+ filename, mode='w+', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def get_reason_list(filepath):
    rea_list = []
    num_list = []
    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            data_pre = eval(lines[i])
            if (data_pre["number"] not in num_list):
                num_list.append(data_pre["number"])
                rea_list.append(data_pre["raw_text"])
    return rea_list, num_list

def merge_reasons(args, rea_list, num_list, filepath, log):
    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()
        for j in range(len(num_list)):
            dic = {}
            reasons = []
            res_list = []
            dic_rea={}
            for i in range(len(lines)):
                data_pre = eval(lines[i])
                if (data_pre["number"] == num_list[j]):
                    prompt = data_pre["prompt"]
                    if (len(data_pre["output"])) != 0:
                        data = data_pre["output"][0]
                        if len(data) != 0:
                            for k in data[args.type]:
                                reasons.append(k)
                    dic_rea[args.type] = reasons
                    res_list.append(data_pre["result_list"])
            dic['raw_text'] = rea_list[j]
            dic['number'] = num_list[j]
            dic['result_list'] = res_list
            dic['prompt'] = prompt
            dic['output'] = [dic_rea]
            log.info(dic)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--type', type=str, default='业绩归因',help='原因类型')
    args = parser.parse_args()

    logpth = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/"
    log = get_logger('merge',logpath)

    # filepath = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/test.txt"
    filepath = "/data/pzy2022/project/eccnlp_local/info_extraction/info_extraction_result_1222.txt"
    rea_list, num_list = get_reason_list(filepath)
    merge_reasons(args, rea_list, num_list, filepath, log)

