import numpy as np
import pandas as pd
import csv
import re
import random
import threading
import logging
import argparse

parser = argparse.ArgumentParser(description='Data Processing')
parser.add_argument('--data', type=str, required=True, help='path of raw data')
parser.add_argument('--length', default=5, type=str, help='minimum length of clause')
parser.add_argument('--type', default='业绩归因', type=str, help='type of answer')
parser.add_argument('--log', type=str, required=True, help='path of log file')
args = parser.parse_args()

def re_pattern():
    Codesdf = pd.read_excel(args.data)

    #仅保留前三类问题以及归因在问题中的样本
    Codesdf['Predict_非业绩归因'] = 0
    Codesdf.loc[(Codesdf['业绩归因问题类型'] != "业绩归因") & (Codesdf['业绩归因问题类型'] !=
                                               "回答特定因素对业绩是否有影响") & (Codesdf['业绩归因问题类型'] != "未来发展趋势"), '是否为业绩归因回答'] = 0
    Codesdf.loc[(Codesdf['原因归属在回答文本中'] == 0) & (Codesdf['是否为业绩归因回答'] == 1), '是否为业绩归因回答'] = 0
    #print("查看业绩归因样本情况\n", Codesdf['是否为业绩归因回答'].value_counts(),"\n")
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(("查看业绩归因样本情况",Codesdf['是否为业绩归因回答'].value_counts()))


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

def get_dataset(data):
    sentence_dict = {}
    sentence_set = set()
    for index, row in data.iterrows():
        # 去除正则过滤的非业绩归因
        if row['Predict_非业绩归因'] == 0:
            Acntet_cut = row['Acntet'].split('。')
            for sentence in Acntet_cut:
                #设置切分后最小长度
                if len(sentence) > args.length:
                # if len(sentence) > 0:
                    #设置问题类型
                    if row['业绩归因问题类型'] == args.type:
                    #if row['业绩归因问题类型'] == '业绩归因' or row['业绩归因问题类型'] == '回答特定因素对业绩是否有影响' or row['业绩归因问题类型'] == '未来发展趋势':
                        for i in range(1,11):
                            if str(row['原因归属' + str(i)]) is None:
                                continue
                            if str(row['原因归属' + str(i)]) in str(sentence):
                                # 去除文本中各种字符，只保留汉字
                                sentence = re.sub('[^\u4e00-\u9fa5]','', str(sentence))
                                sentence_dict[sentence] = 1
                                sentence_set.add(sentence)
            
                    if sentence not in sentence_set:
                        sentence = re.sub('[^\u4e00-\u9fa5]','', str(sentence))
                        sentence_dict[sentence] = 0

    train_list = []
    val_list = []
    test_list = []
    train_list1 = []

    # 将text和label合并
    text = list(sentence_dict.keys())
    label = list(sentence_dict.values())
    sentences = []
    for i in range(len(sentence_dict)):
        sentences.append(text[i]+ '\t'+ str(label[i]))

    #分割数据集
    for i in range(len(sentences)):
        if i % 5 ==0:
            test_list.append(sentences[i])
        if i % 5 ==1:
            val_list.append(sentences[i])
        else:
            if sentences[i][-1] =='1':
                train_list1.append(sentences[i])
            train_list.append(sentences[i])

    #均衡样本数据
    train_num = len(train_list)
    num_1 = len(train_list1)
    train_balance_list = [i for i in train_list]

    for i in range(train_num - num_1):
        j = np.random.randint(0,num_1)
        train_balance_list.append(train_list[j])
    random.shuffle(train_balance_list)

    data_train=pd.DataFrame(train_balance_list)
    data_val=pd.DataFrame(val_list)
    data_test=pd.DataFrame(test_list)

    logging.info('问题类型：'+ args.type)
    logging.info('训练集数量：' + str(len(train_list)))
    logging.info('验证集数量：' + str(len(val_list)))
    logging.info('测试集数量：' + str(len(test_list)))
    logging.info('训练集均衡后数量：' + str(len(train_balance_list)))

    return sentence_dict, data_train, data_val, data_test
    
if __name__ == '__main__':
    data = re_pattern()
    get_dataset(data)