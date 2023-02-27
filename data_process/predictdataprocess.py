import re
import random
import logging
import argparse
import numpy as np
import pandas as pd
import time

# parser = argparse.ArgumentParser(description='Data Processing')
# parser.add_argument('--data', type=str, required=True, help='path of raw data')
# parser.add_argument('--min_length', default=5, type=int, help='longer than 0')
# parser.add_argument('--balance', default='none',type=str, required=True, help='up or down or none')
# args = parser.parse_args()

'''正则过滤部分非业绩归因问题'''
def re_pattern1(args):
    Codesdf = pd.read_excel(args.data)
    Codesdf['Predict_非业绩归因'] = 0
    log_filename = './log/' + '3.1_'+time.strftime('%m.%d_%H.%M', time.localtime()) + '.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

    def Prediction(Condition):
        Len1 = len(Codesdf[Codesdf['Predict_非业绩归因'] == 1])
        logging.info("增加{}后过滤数量:{}".format(Condition, Len1))

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

    pattern4= "未来|计划|将|预计|下一年"
    Codesdf.loc[(Codesdf['Qcntet'].str.contains(pattern4) == 1) & (
        Codesdf['Acntet'].str.contains(Attr_pattern) == 0), 'Predict_非业绩归因'] = 1
    # Codesdf.loc[(Codesdf['Qcntet'].str.contains(pattern4) == 1) , 'Predict_非业绩归因'] = 1
    Prediction("未来规划")

    pattern5= "(没有|暂无)[^\r\n\t\f\v，,。？！]*影响"
    Codesdf.loc[(Codesdf['Qcntet'].str.contains(pattern5) == 1) , 'Predict_非业绩归因'] = 1
    Prediction("非归因")

    len1 = len(Codesdf[Codesdf['Predict_非业绩归因'] == 1])
    logging.info("共过滤非业绩归因回答数量："+str(len1))
    return Codesdf

def re_pattern2(args):
    Codesdf = pd.read_excel(args.data)
    Codesdf['Predict_非业绩归因'] = 0
    log_filename = './log/' + '3.2_'+time.strftime('%m.%d_%H.%M', time.localtime()) + '.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

    def Prediction(Condition):
        Len1 = len(Codesdf[Codesdf['Predict_非业绩归因'] == 1])
        logging.info("增加{}后过滤数量:{}".format(Condition, Len1))

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

    pattern4= "未来|计划|将|预计|下一年"
    # Codesdf.loc[(Codesdf['Qcntet'].str.contains(pattern4) == 1) & (
    #     Codesdf['Acntet'].str.contains(Attr_pattern) == 0), 'Predict_非业绩归因'] = 1
    Codesdf.loc[(Codesdf['Qcntet'].str.contains(pattern4) == 1) , 'Predict_非业绩归因'] = 1
    Prediction("未来规划")

    pattern5= "(没有|暂无)[^\r\n\t\f\v，,。？！]*影响"
    Codesdf.loc[(Codesdf['Qcntet'].str.contains(pattern5) == 1) , 'Predict_非业绩归因'] = 1
    Prediction("非归因")

    len1 = len(Codesdf[Codesdf['Predict_非业绩归因'] == 1])
    logging.info("共过滤非业绩归因回答数量："+str(len1))
    
    return Codesdf

# 生成预测数据集
def predict_dataset(data, args):
    predict_data = data
    dataset_list = []
    for index, row in predict_data.iterrows():
        # 去除正则过滤的非业绩归因回答
        if row['Predict_非业绩归因'] == 0:
            split_rule = '[。？！]'
            Acntet_cut = re.split(split_rule,row['Acntet'])
            for sentence in Acntet_cut:
                sentence = re.sub('\\n|\?|&lt;|br|&gt;','', str(sentence))
                #设置切分后最小长度
                if len(sentence) > args.min_length:
                    data_dict = {'number':'', 'content': '', 'raw_text':'', 'label': 0, 'result_list': [{"text": '', "start":None , "end":None}], 'prompt': ''}
                    data_dict['number'] = row['No']
                    data_dict['content'] = sentence
                    data_dict['raw_text'] = row['Acntet']
                    dataset_list.append(data_dict)
    logging.info('预测数据总量：'+ str(len(dataset_list)))
    return dataset_list

# 将dict转换为list
def dict_to_list(data_dict):
    data_list = []
    if len(data_dict) > 0:
        for i in range(len(data_dict)):
            content, label = data_dict[i]['content'],data_dict[i]['label']
            data_list.append(list((content,label)))
    return data_list
