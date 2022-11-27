import re
import random
import logging
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Data Processing')
parser.add_argument('--data', type=str, required=True, help='path of raw data')
parser.add_argument('--min_length', default=5, type=str, help='not less than 0')
parser.add_argument('--type', default='业绩归因', type=str, help='type of answer')
parser.add_argument('--balance', default='none',type=str, required=True, help='up or down or none')
# parser.add_argument('--predict', type=str, required=True, help='path of predict data')
args = parser.parse_args()

def re_pattern(args):
    Codesdf = pd.read_excel(args.data)

    #仅保留前三类问题以及归因在问题中的样本
    Codesdf['Predict_非业绩归因'] = 0
    Codesdf.loc[(Codesdf['业绩归因问题类型'] != "业绩归因") & (Codesdf['业绩归因问题类型'] !=
                                               "回答特定因素对业绩是否有影响") & (Codesdf['业绩归因问题类型'] != "未来发展趋势"), '是否为业绩归因回答'] = 0
    Codesdf.loc[(Codesdf['原因归属在回答文本中'] == 0) & (Codesdf['是否为业绩归因回答'] == 1), '是否为业绩归因回答'] = 0
    #print("查看业绩归因样本情况\n", Codesdf['是否为业绩归因回答'].value_counts(),"\n")
    logging.basicConfig(filename='./log/preprocess.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(("查看业绩归因样本情况",(Codesdf['是否为业绩归因回答'].value_counts())))


    # 定义函数方便查看筛选效果
    def Prediction(Condition):
        Length=len(Codesdf[(Codesdf['是否为业绩归因回答'] == 0)])
        Len1 = len(Codesdf[Codesdf['Predict_非业绩归因'] == 1])
        Len2 = len(Codesdf[(Codesdf['Predict_非业绩归因'] == 1)
               & (Codesdf['是否为业绩归因回答'] == 1)])
        logging.info(("总非业绩归因条目数", Length))
        logging.info(("增加%s后过滤数量:%d" %(Condition, Len1)))
        logging.info(("其中错误过滤数量", Len2))
        logging.info(("其中正确过滤数量", Len1-Len2))

    # 定义业绩归因表达式
    Attr_pattern = '业绩|利润|亏损|损失|收入|绩效|竞争力|成本|盈利|影响|利于|发展|竞争力'  # 放宽标准则仅包括'影响|利于|发展|竞争力'

    # 拒不回答
    Codesdf['Length'] = Codesdf['Acntet'].apply(lambda x: len(x))
    Codesdf.loc[(Codesdf['Acntet'].str.contains('感谢|公告|报告|披露|回复|回答') == 1)
            & (Codesdf['Length'] <= 30), 'Predict_非业绩归因'] = 1
    Codesdf.loc[(Codesdf['Length'] < 20) & (
        Codesdf['Acntet'].str.contains(Attr_pattern) == 0), 'Predict_非业绩归因'] = 1
    Prediction("拒不回答类型")

    # 无关问题：股市、薪酬、信息披露
    pattern1 = '分红|派息|股权|利润分配|股份|股价|股市|市胆率|PE|不良率|大盘'
    Codesdf.loc[(Codesdf['Qcntet'].str.contains(pattern1) == 1) & (
        Codesdf['Acntet'].str.contains(Attr_pattern) == 0), 'Predict_非业绩归因'] = 1
    Codesdf.loc[(Codesdf['Qcntet'].str.contains('薪酬|信息披露') == 1), 'Predict_非业绩归因'] = 1
    Prediction("无关问题")

    # 答非所问
    Codesdf.loc[(Codesdf['Acntet'].str.contains('感谢|公告|报告|披露|回复|回答') == 1)
            & (Codesdf['Length'] <= 30), 'Predict_非业绩归因'] = 1

    Codesdf.loc[(Codesdf['Length'] < 20) & (
        Codesdf['Acntet'].str.contains(Attr_pattern) == 0), 'Predict_非业绩归因'] = 1

    pattern3 = '不*回答问题|回避问题|官话|答非所问|没有回[复答]|态度问题'
    Codesdf.loc[(Codesdf['Qcntet'].str.contains(pattern3) == 1) & (
        Codesdf['Acntet'].str.contains(Attr_pattern) == 0), 'Predict_非业绩归因'] = 1
    Prediction("答非所问")

    #输出结果
    Codesdf=Codesdf.drop(columns=['Length'])
    return Codesdf

# 模型训练数据
def train_dataset(data, args):
    dataset_list = []
    sentence_set = set()
    for index, row in data.iterrows():
        # 去除正则过滤的非业绩归因回答
        if row['Predict_非业绩归因'] == 0:
            split_rule = '[。？！]'
            Acntet_cut = re.split(split_rule,row['Acntet'])
            for sentence in Acntet_cut:
                #设置切分后最小长度
                if len(sentence) > args.min_length:
                    data_dict = {'content': '', 'raw_text':'', 'label': 0, 'result_list': [{"text": '', "start":None , "end":None}], 'prompt': ''}
                    data_dict['content'] = sentence
                    data_dict['raw_text'] = row['Acntet']
                    result_dict = {}
                    #设置问题类型
                    if row['业绩归因问题类型'] == args.type:
                        for i in range(1,11):
                            if str(row['原因归属' + str(i)]) is None:
                                continue
                            if str(row['原因归属' + str(i)]) in str(sentence):
                                data_dict['label'] = 1
                                result_dict['text'] = str(row['原因归属' + str(i)])
                                result_dict['start'] = sentence.index(str(row['原因归属' + str(i)]))
                                result_dict['end'] = result_dict['start'] + len(row['原因归属' + str(i)])
                                sentence_set.add(sentence)

                    if sentence not in sentence_set:
                        data_dict['label'] = 0
                    data_dict['result_list'] = [result_dict]
                    data_dict['prompt'] = args.type
                dataset_list.append(data_dict)
    # logging.info((dataset_list))
    return dataset_list


# 模型预测数据
def predict_dataset(args):
    predict_data = pd.read_excel(args.predict)
    dataset_list = []
    for index, row in predict_data.iterrows():
        # 去除正则过滤的非业绩归因回答
        if row['Predict_非业绩归因'] == 0:
            split_rule = '[。？！]'
            Acntet_cut = re.split(split_rule,row['Acntet'])
            for sentence in Acntet_cut:
                #设置切分后最小长度
                if len(sentence) > args.min_length:
                    data_dict = {'content': '', 'raw_text':'', 'label': None, 'result_list': [{"text": '', "start":None , "end":None}], 'prompt': ''}
                    data_dict['content'] = sentence
                    data_dict['raw_text'] = row['Acntet']
                dataset_list.append(data_dict)
    return dataset_list

# 生成分类模型数据集
def classification_dataset(dataset,args):
    train_list = []
    val_list = []
    test_list = []
    train_list0 = []
    train_list1 = []
    val_list0 = []
    val_list1 = []


    #分割数据集(1:1:3)
    for i in range(len(dataset)):
        if i % 5 == 0:
            content, label = dataset[i]['content'], dataset[i]['label']
            test_list.append(list((content, label)))
        elif i % 5 == 1:
            content, label = dataset[i]['content'], dataset[i]['label']
            val_list.append(list((content, label)))
            if label == 0:
                val_list0.append(list((content, label)))
            else:
                val_list1.append(list((content, label)))
        else:
            content, label = dataset[i]['content'], dataset[i]['label']
            train_list.append(list((content, label)))
            if label == 0:
                train_list0.append(list((content, label)))
            else:
                train_list1.append(list((content, label)))
            
    # 获取训练集、验证集样本数
    train_num = len(train_list)
    val_num = len(val_list)
    train_num1 = len(train_list1)   
    val_num1 = len(val_list1)


    train_balance_list = []
    val_balance_list = []
    # 对训练集、测试集进行上采样
    def upsampling():
        for i in range(train_num - train_num1):
            j = np.random.randint(0, train_num1)
            train_balance_list.append(train_list1[j])
        for i in range(val_num - val_num1):
            j = np.random.randint(0, val_num1)
            val_balance_list.append(val_list1[j])
        random.shuffle(train_balance_list)
        random.shuffle(val_balance_list)

    # 对训练集、测试集进行下采样
    def downsampling():
        train_num_list = np.random.randint(train_num-train_num1, size=train_num1)
        val_num_list = np.random.randint(val_num-val_num1, size=val_num1)
        for i in range(train_num1):
            train_balance_list.append(train_list0[train_num_list[i]])
        for i in range(val_num1):
            train_balance_list.append(train_list0[val_num_list[i]])
        random.shuffle(train_balance_list)
        random.shuffle(val_balance_list)      


    logging.info('问题类型：'+ args.type)
    logging.info('训练集数量：' + str(len(train_list)))
    logging.info('验证集数量：' + str(len(val_list)))
    logging.info('测试集数量：' + str(len(test_list)))

    train_df = pd.DataFrame(train_list)
    val_df = pd.DataFrame(val_list)
    test_df = pd.DataFrame(test_list)

    if args.balance != 'none':
        if args.balance == 'up':
            upsampling()
        if args.balance == 'down':
            downsampling()
        logging.info('训练集均衡后数量：' + str(len(train_balance_list)))
        logging.info('验证集均衡后数量：' + str(len(val_balance_list)))
        train_df = pd.DataFrame(train_balance_list)
        val_df = pd.DataFrame(val_balance_list)
    
    return train_df, val_df, test_df