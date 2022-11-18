import numpy as np
import pandas as pd
import csv
import re
import random
import threading

def re_pattern():
    Codesdf = pd.read_excel('/data/xf2022/Projects/TextClassificationModel/data/raw_data/第一期手工标注工作核查结果-1113.xlsx')

    #仅保留前三类问题以及归因在问题中的样本
    Codesdf['Predict_非业绩归因'] = 0
    Codesdf.loc[(Codesdf['业绩归因问题类型'] != "业绩归因") & (Codesdf['业绩归因问题类型'] !=
                                               "回答特定因素对业绩是否有影响") & (Codesdf['业绩归因问题类型'] != "未来发展趋势"), '是否为业绩归因回答'] = 0
    Codesdf.loc[(Codesdf['原因归属在回答文本中'] == 0) & (Codesdf['是否为业绩归因回答'] == 1), '是否为业绩归因回答'] = 0
    print("查看业绩归因样本情况\n", Codesdf['是否为业绩归因回答'].value_counts(),"\n")

    # 定义函数方便查看筛选效果
    def Prediction(Condition):
        Length=len(Codesdf[(Codesdf['是否为业绩归因回答'] == 0)]  )
        Len1 = len(Codesdf[Codesdf['Predict_非业绩归因'] == 1])
        Len2 = len(Codesdf[(Codesdf['Predict_非业绩归因'] == 1)
               & (Codesdf['是否为业绩归因回答'] == 1)])
        print("总非业绩归因条目数", Length)
        print("增加%s后过滤数量:%d" %(Condition, Len1))
        print("其中错误过滤数量", Len2)
        print("其中正确过滤数量", Len1-Len2)
        print("\n")

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
    Codesdf.to_csv('./data/raw_data/test.csv')

def get_dataset():
    data = pd.read_csv('./data/raw_data/test.csv')
    print(data.head())
    data.info()
    sentence_dict = {}
    sentence_set = set()
    for index, row in data.iterrows():
        # 去除正则过滤的非业绩归因
        if row['Predict_非业绩归因'] == 0:
            Acntet_cut = row['Acntet'].split('。')
            for sentence in Acntet_cut:
                # 去除文本中各种字符，只保留汉字
                sentence = re.sub('[^\u4e00-\u9fa5]','', str(sentence))
                #设置切分后最小长度
                if len(sentence) > 5:
                # if len(sentence) > 0:
                    #设置问题类型
                    if row['业绩归因问题类型'] == '业绩归因':
                    #if row['业绩归因问题类型'] == '业绩归因' or row['业绩归因问题类型'] == '回答特定因素对业绩是否有影响' or row['业绩归因问题类型'] == '未来发展趋势':
                        for i in range(1,11):
                            if str(row['原因归属' + str(i)]) is None:
                                continue
                            if str(row['原因归属' + str(i)]) in str(sentence):
                                sentence_dict[sentence] = 1
                                sentence_set.add(sentence)
            
                    if sentence not in sentence_set:
                        sentence_dict[sentence] = 0
        
    with open('./data/raw_data/acntet_cut_test.csv','w',encoding='utf-8', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['Acntet','是否为业绩归因回答'])
        for k in sentence_dict.keys():
            f_csv.writerow([k, sentence_dict[k]])
    f.close()


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

    data_train = open('/data/xf2022/Projects/TextClassificationModel/data/dataset_test/train.txt','w')
    data_test = open('/data/xf2022/Projects/TextClassificationModel/data/dataset_test/test.txt', 'w', encoding='utf8')
    data_val = open('/data/xf2022/Projects/TextClassificationModel/data/dataset_test/validation.txt', 'w', encoding='utf8')
    data_description = open('/data/xf2022/Projects/TextClassificationModel/data/dataset_test/description.txt', 'w', encoding='utf8')

    for sentence in train_balance_list:
        data_train.write(sentence + '\n')
    data_train.close()
    print('训练集写入完成。')

    for sentence in val_list:
        data_val.write(sentence + '\n')
    data_val.close()
    print('验证集写入完成。')

    for sentence in test_list:
        data_test.write(sentence + '\n')
    data_test.close()
    print('测试集写入完成。')

    data_description.write('切分不设阈值'+'\n'+'问题类型：业绩归因+回答特定因素对业绩是否有影响+未来发展趋势'+'\n')
    data_description.write('训练集数量：' + str(len(train_list)) +'\n')
    data_description.write('验证集数量：' + str(len(val_list)) +'\n')
    data_description.write('测试集数量：' + str(len(test_list)) + '\n')
    data_description.write('训练集均衡后数量：' + str(len(train_balance_list)) +'\n')
    data_description.close()
    
if __name__ == '__main__':
    re_pattern()
    get_dataset()