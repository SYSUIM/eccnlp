import numpy as np
import csv
import pandas as pd
import logging
import math
import argparse
from datetime import datetime
from rank_data_process import print_list, read_list

def get_logger(name):
    logger = logging.getLogger(name)
    filename = f'{datetime.now().date()}_{name}.txt'
    fh = logging.FileHandler("./data/res_log/2.0_"+ filename, mode='w+', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_reason_list(uie_list):
    rea_list = []
    num_list = []
    lines = uie_list      
    for i in range(len(lines)):
        data_pre = eval(lines[i])
        if (data_pre["number"] not in num_list):
            num_list.append(data_pre["number"])
            rea_list.append(data_pre["raw_text"])
    return rea_list, num_list


def merge_reasons(args, rea_list, num_list, uie_list):
    lines = uie_list
    merged_reasons = []
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
        merged_reasons.append(dic)
    return merged_reasons


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='merge reasons')
    parser.add_argument('--type', type=str, default='业绩归因',help='原因类型')
    args = parser.parse_args()

    filepath = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/test.txt"
    # filepath = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2.0_2023-01-14_add_embedding.txt"
    # filepath1 = "/data/pzy2022/project/eccnlp_local/info_extraction/info_extraction_result_0113_test.txt"
    filepath2 = "/data/pzy2022/project/eccnlp_local/info_extraction/info_extraction_result_0113.txt"

    log=get_logger('merge_test')

    uie_list = read_list(filepath2)
    rea_list, num_list = get_reason_list(uie_list)
    merged_reasons = merge_reasons(args, rea_list, num_list, uie_list)
    print_list(merged_reasons)



