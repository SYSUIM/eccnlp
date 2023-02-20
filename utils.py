import os
import logging
import numpy as np
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
    logging.info(f'read data_list DONE. Length: {len(data_list)}')

    return data_list

'''生成分类模型数据集'''
def split_dataset(dataset,args):
    content_list = []
    label_list = []

    for i in range(len(dataset)):
        content, label = dataset[i]['content'],dataset[i]['label']
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

if __name__ == '__main__':
    path = '/data/pzy2022/project/eccnlp/data_process/after_classification_data3.1.txt'
    read_list_file(path)