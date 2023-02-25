import os
import re
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(
    level=logging.INFO,
     format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
     datefmt='%Y-%m-%d %H:%M:%S'
     )

def read_list_file(path: str) -> list:
    data_list = []
    with open(path, 'r') as f:
        for line in f:
            data_list.append(eval(line.strip('\n'), {'nan': ''}))
    logging.info(f'read data_list DONE. Length of {path}: {len(data_list)}')

    return data_list

def evaluate(true_list: list, predict_list: list):
    predict_index_list, true_index_list = {}, []
    for predict in predict_list:
        predict_index_list[predict['number']] = predict['rerank']

    for true in true_list:
        true_index_list.append(true['number'])

    # ensure unduplicated list be created, otherwise exit
    if len(predict_index_list) != len(set(predict_index_list)):
        print('len(predict_index_list):', len(predict_index_list))
        print('len(set(predict_index_list)):', len(set(predict_index_list)))
        exit(0)

    proceseed_num = []
    list1, list2 = [], []
    for true in true_list:
        index = true['number']
        if index in proceseed_num:
            continue
        if index in predict_index_list.keys():
            true_flag, predict_flag = None, None
            proceseed_num.append(index)
            if not true['prompt']:
                true_flag = 'negetive'
                if not predict_index_list[index][0]:
                    predict_flag = 'negetive'
                else:
                    predict_flag = 'positive'
            else:
                true_flag = 'positive'
                for reason in true['result_list']:
                    if (str(predict_index_list[index][0]) in reason['text']) or (reason['text'] in str(predict_index_list[index][0])):
                        predict_flag = 'positive'
                        break
                if predict_flag is None:
                    predict_flag = 'negetive'
            list1.append(true_flag)
            list2.append(predict_flag)

    classification_report_actual = classification_report(list1, list2)
    logging.info(f'\n{classification_report_actual}')


'''生成分类模型数据集'''
def split_dataset(dataset,args):
    '''
    Split dataset into train, validation(if you want) and test.
    The parameters could be defined in args(initalized in config.py):
        1. test_size: the size of test dataset. 
        2. val_size: the size of validation dataset.
        3. train_size: the size of train dataset.
        NOTICE: The sum of three above needs to be 1, otherwise train_size = 1 - test_size - val_dataset. 
    '''
    content_list = []
    label_list = []

    for i in range(len(dataset)):
        content, label = dataset[i]['content'], dataset[i]['label']
        content = re.sub("[^\u4e00-\u9fa5]", "", str(content))
        if content:           # 去除只保留汉字后出现的空值
            content_list.append(content)
            label_list.append(label)
    
    X = np.array(content_list)
    y = np.array(label_list)
    if args.test_size > 0:
        # 划分训练集和测试集       
        split = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=4)
        for train_index, test_index in split.split(X, y):
            train_num = train_index 
            test_num = test_index
        if args.val_size > 0:
             #划分验证集    
            split = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size/(1-args.test_size), random_state=4)
            for train_index, val_index in split.split(X[train_num], y[train_num]):
                train_tmp = train_num[train_index]
                val_num = train_num[val_index]
                train_num = train_tmp
    else:        
        #无测试集，只有训练集和验证集
        split = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=4)
        for train_index, val_index in split.split(X,y):
            train_num = train_index
            val_num = val_index
    train_dict = []
    val_dict = []
    test_dict = []
    train_dict = [dataset[i] for i in train_num]
    if args.val_size > 0:
        val_dict = [dataset[i] for i in val_num]
    if args.test_size > 0:
        test_dict = [dataset[i] for i in test_num]
    
    return train_dict, val_dict, test_dict

def evaluate_sentence(result_list, classification_list):
    predict, true = [], []
    sentence_number = set()

    for sample in classification_list:
        if sample['label'] == 1:
            sentence_number.add(sample['number'])

    for data in result_list:
        if data['number'] not in sentence_number:
            true.append(0)
            predict.append(0)
            continue

        if data['label'] == 0:
            true.append(0)
        else:
            true.append(1)

        if data['output'][0]:
            predict.append(0)
        else:
            predict.append(1)
        
    classification_report_actual = classification_report(true, predict)
    logging.info(f'\n{classification_report_actual}')
    

if __name__ == '__main__':
    # test for read_list_file
    # path = '/data/pzy2022/project/eccnlp/data_process/after_classification_data3.1.txt'
    # read_list_file(path)

    # test for evaluate
    # true_path = '/data/pzy2022/project/eccnlp_local/2.0_raw_dict.txt'
    # predict_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2023-01-15_predict-M15-0113_test.txt'
    # evaluate(read_list_file(true_path), read_list_file(predict_path))

    # test for evaluate sentence
    result_list = read_list_file('/data/pzy2022/project/eccnlp/info_extraction/after_extraction_data3.1.txt')
    classification_list = read_list_file('/data/pzy2022/project/eccnlp/data_process/after_classification_data3.1.txt')
    evaluate_sentence(result_list, classification_list)
    pass