import re
import random
import logging
import numpy as np
import pandas as pd
import time
import jieba
from sklearn.model_selection import StratifiedShuffleSplit

# parser = argparse.ArgumentParser(description='Data Processing')
# parser.add_argument('--data', type=str, required=True, help='path of raw data')
# parser.add_argument('--min_length', default=5, type=str, help='not less than 0')
# parser.add_argument('--train_size', default=0.6, type=float, help='ratio of train data')
# parser.add_argument('--val_size', default=0.2, type=float, help='ratio of train data')
# parser.add_argument('--test_size', default=0.2, type=float, help='ratio of train data')
# parser.add_argument('--balance', default='none',type=str, required=True, help='up or down or none')
# parser.add_argument('--predict_data', type=str, required=True, help='path of predict data')
# args = parser.parse_args()

'''正则过滤部分非业绩归因问题'''
def re_pattern1(args):
    Codesdf = pd.read_excel(args.data)
    Codesdf['Predict_非业绩归因'] = 0
    log_filename = './log/' + '2.1_'+args.balance+'_'+time.strftime('%m.%d_%H.%M', time.localtime()) + '.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
    logging.info("查看原始业绩归因样本情况:\n{}\n".format(Codesdf['是否为业绩归因回答'].value_counts()))
    # Codesdf.loc[(Codesdf['业绩归因问题类型'] != "业绩归因") & (Codesdf['业绩归因问题类型'] != "回答特定因素对业绩是否有影响") & (Codesdf['业绩归因问题类型'] != "分析优劣势"), '是否为业绩归因回答'] = 0
    Codesdf.loc[(Codesdf['原因归属在回答文本中'] == 0) & (Codesdf['是否为业绩归因回答'] == 1), '是否为业绩归因回答'] = 0
    logging.info("查看归因样本情况:\n{}\n".format(Codesdf['是否为业绩归因回答'].value_counts()))


    # 定义函数方便查看筛选效果
    def Prediction(Condition):
        # Length=len(Codesdf[(Codesdf['是否为业绩归因回答'] == 0)])
        Len1 = len(Codesdf[Codesdf['Predict_非业绩归因'] == 1])
        # logging.info("总非业绩归因条目数", Length)
        logging.info("增加{}后过滤数量:{}\n".format(Condition, Len1))


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

    Len1 = len(Codesdf[Codesdf['Predict_非业绩归因'] == 1])
    Len2 = len(Codesdf[(Codesdf['Predict_非业绩归因'] == 1)
               & (Codesdf['是否为业绩归因回答'] == 1)])
    logging.info("共过滤非业绩归因回答数量："+ str(Len1))
    logging.info("其中错误过滤数量：" + str(Len2))
    logging.info("其中正确过滤数量："+ str(Len1-Len2))
    logging.info("Precision："+ str(round((Len1-Len2)/Len1,4)))

    return Codesdf

def re_pattern2(args):
    Codesdf = pd.read_excel(args.data)
    Codesdf['Predict_非业绩归因'] = 0
    log_filename = './log/' + '2.2_'+args.balance+'_'+time.strftime('%m.%d_%H.%M', time.localtime()) + '.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
    logging.info("查看原始业绩归因样本情况:\n{}".format(Codesdf['是否为业绩归因回答'].value_counts()))
    # Codesdf.loc[(Codesdf['业绩归因问题类型'] != "业绩归因") & (Codesdf['业绩归因问题类型'] != "回答特定因素对业绩是否有影响") & (Codesdf['业绩归因问题类型'] != "分析优劣势"), '是否为业绩归因回答'] = 0
    Codesdf.loc[(Codesdf['原因归属在回答文本中'] == 0) & (Codesdf['是否为业绩归因回答'] == 1), '是否为业绩归因回答'] = 0
    logging.info("查看归因样本情况:\n{}".format(Codesdf['是否为业绩归因回答'].value_counts()))


    # 定义函数方便查看筛选效果
    def Prediction(Condition):
        # Length=len(Codesdf[(Codesdf['是否为业绩归因回答'] == 0)])
        Len1 = len(Codesdf[Codesdf['Predict_非业绩归因'] == 1])
        # logging.info("总非业绩归因条目数", Length)
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

    Len1 = len(Codesdf[Codesdf['Predict_非业绩归因'] == 1])
    Len2 = len(Codesdf[(Codesdf['Predict_非业绩归因'] == 1)
               & (Codesdf['是否为业绩归因回答'] == 1)])
    logging.info("总共过滤非业绩归因回答数量："+ str(Len1))
    logging.info("其中错误过滤数量：" + str(Len2))
    logging.info("其中正确过滤数量："+ str(Len1-Len2))
    logging.info("Precision："+ str(round((Len1-Len2)/Len1,4)))

    Codesdf=Codesdf.drop(columns=['Length'])
    return Codesdf


'''模型训练数据'''
def train_dataset(data, args):
    dataset_list = []
    for index, row in data.iterrows():
        # 去除正则过滤的非业绩归因回答
        if row['Predict_非业绩归因'] == 0:
            split_rule = '[。？！]'
            Acntet_cut = re.split(split_rule,row['Acntet'])
            for sentence in Acntet_cut:
                sentence = re.sub('\\n|\?|&lt;|br|&gt;','', str(sentence))
                #设置切分后最小长度
                if len(sentence) > int(args.min_length):
                    data_dict = {'number':'','content': '', 'raw_text':'', 'label': 0, 'result_list': [{"text": '', "start":None , "end":None}], 'prompt': ''}
                    data_dict['number'] = row['No']
                    data_dict['content'] = sentence
                    data_dict['raw_text'] = row['Acntet']
                    result_dict = {}
                    #设置问题类型
                    # if row['业绩归因问题类型'] == '业绩归因':
                    if row['业绩归因问题类型'] == '业绩归因' or row['业绩归因问题类型'] == '回答特定因素对业绩是否有影响' or row['业绩归因问题类型'] == '未来发展趋势':
                        for i in range(1,11):
                            if str(row['原因归属' + str(i)]) is None:
                                continue
                            if str(row['原因归属' + str(i)]) in str(sentence):
                                data_dict['label'] = 1
                                result_dict['text'] = str(row['原因归属' + str(i)])
                                result_dict['start'] = sentence.index(str(row['原因归属' + str(i)]))
                                result_dict['end'] = result_dict['start'] + len(row['原因归属' + str(i)])
                    data_dict['result_list'] = [result_dict]
                    data_dict['prompt'] = row['业绩归因问题类型']
                    dataset_list.append(data_dict)
    logging.info('数据集总量：'+ str(len(dataset_list)))
    return dataset_list


''' 模型预测数据 '''
def predict_dataset(args):
    predict_data = pd.read_excel(args.predict_data)
    dataset_list = []
    for index, row in predict_data.iterrows():
        # 去除正则过滤的非业绩归因回答
        if row['Predict_非业绩归因'] == 0:
            split_rule = '[。？！]'
            Acntet_cut = re.split(split_rule,row['Acntet'])
            for sentence in Acntet_cut:
                sentence = re.sub('[\n,?]','', str(sentence))
                #设置切分后最小长度
                if len(sentence) > args.min_length:
                    data_dict = {'number':'', 'content': '', 'raw_text':'', 'label': 0, 'result_list': [{"text": '', "start":None , "end":None}], 'prompt': ''}
                    data_dict['number'] = row['No']
                    data_dict['content'] = sentence
                    data_dict['raw_text'] = row['Acntet']
            dataset_list.append(data_dict)
    return dataset_list


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


'''对数据集进行采样'''
def sampling(data_dict,args):
    for i in range(len(data_dict)):
        data_dict0 = []
        data_dict1 = []
        # 记录正负样本数
        for i in range(len(data_dict)):
            if data_dict[i]['label'] == 0:
                data_dict0.append(data_dict[i])
            else:
                data_dict1.append(data_dict[i])
    data_num1 = len(data_dict1)
    data_num0 = len(data_dict0)

    def upsampling():
        i = 1
        while i*data_num1 <= data_num0-data_num1:
            data_dict.extend(data_dict1)
            i+=1
        data_num = data_num0 - i*data_num1
        random.seed(10)
        data_num_list = random.sample(range(0,data_num1),data_num)
        for i in range(data_num):
            data_dict.append(data_dict1[data_num_list[i]])
    
    def downsampling():
        random.seed(10)
        data_num_list = random.sample(range(0, data_num0), data_num1)
        for i in range(data_num1):
            data_dict.append(data_dict0[data_num_list[i]])

    if args.balance == 'up':
        data_dict = [i for i in data_dict0]
        upsampling()
    if args.balance == 'down':
        data_dict = [i for i in data_dict1]
        downsampling()
    return data_dict

'''将dict转换为list'''
def dict_to_list(data_dict):
    data_list = []
    for i in range(len(data_dict)):
        content, label = data_dict[i]['content'],data_dict[i]['label']
        data_list.append(list((content,label)))
    return data_list


'''生成分类数据集并选择采样方法'''
def classification_dataset(train_dict,val_dict,test_dict,args):
    if args.balance != 'none':
        train_dict = sampling(train_dict,args)
        if args.val_size > 0:
            val_dict = sampling(val_dict,args)

    train_list = dict_to_list(train_dict)
    val_list = dict_to_list(val_dict)
    test_list = dict_to_list(test_dict)

    random.seed(10)
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    random.shuffle(train_dict)
    random.shuffle(val_dict)
    random.shuffle(test_dict)

    logging.info('采样方式：'+ args.balance)
    logging.info('训练集数量：' + str(len(train_dict)))
    logging.info('验证集数量：' + str(len(val_dict)))
    logging.info('测试集数量：' + str(len(test_dict)))

    return train_list, val_list, test_list, train_dict, val_dict, test_dict


'''传入dataset_list构建词库''' 
def build_thesaurus(dataset, t_path):
    '''加载停用词'''
    with open(t_path) as fr:
        stop_words = set([word.strip() for word in fr])

    all_label_pro = [] # 保存标注的原因
    for i in range(len(dataset)):
        if dataset[i]['label'] == 1:
            line = dataset[i]
            data = line['result_list'][0]['text']
            if data != '':
                all_label_pro.append(data)
    
    # 将标注的原因归属分割成词或词语
    # mylog = open(w_path,mode='w',encoding='utf-8')
    words = []
    word =[]
    for i in range(len(all_label_pro)):
        all_label_pro[i] = jieba.lcut(str(all_label_pro[i]))
        words += all_label_pro[i]
    for i in words:
        if i not in stop_words:
            word.append(i)
    return word