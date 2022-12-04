import functools
import difflib
import random
import numpy as np
import csv
import logging
import math
import config
import operator
import argparse
from datetime import datetime

def get_logger(name):
    logger = logging.getLogger(name)
    filename = f'{datetime.now().date()}_{name}.log'
    fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

#定义两个两个原因的排序标准
#第一维：相似度 1
#第二维：概率 2
#第三维：词出现的次数 3
def list_cmp(x,y):
    if x[1] != y[1]:
        return x[1]>y[1]
    elif x[3] != y[3]:
        return x[3]>y[3]
    else:
        return x[2]>y[2]

#归一化
def normalize(a_list):
    y=[]
    arr = np.asarray(a_list)
    for x in arr:
        x = float(x - np.min(arr))/(np.max(arr)- np.min(arr))
        y.append(x)
    return y

#计算词库中的词在原因中出现的次数cot
def calculate_cot(j,vocab):
    cot=0
    for search_list in vocab:
        if search_list in j["text"]:
            cot=cot+1
    return cot

# 排序后生成标签，rank1标记label为1，其余为-1              
def generate_label(line_elem,all_rows):
    line_elem.sort(key=functools.cmp_to_key(list_cmp))
    for j in range(len(line_elem)):
        if j==0:
            line_elem[j].append(1)
        else:
            line_elem[j].append(-1)
        all_rows.append(line_elem[j])
    return all_rows

#为每个UIE提取原因构造input格式的向量
def form_input_vector(all_rows,cnt):
    y=[] 
    list_len=len(all_rows)
    for i in range(list_len):
        line_in_all=[]
        line_in_all.append(all_rows[i][4]) #label
        line_in_all.append(all_rows[i][0]) #key_num
        line_in_all.append(all_rows[i][2]) #probability
        line_in_all.append(cnt[i])         #词库次数cot(归一化后）
        y.append(line_in_all)
    return y

#加入构建的负样本
def generate_negative_sample_and_addin(all,list_len):
    alllist=[]
    f_num=5
    i=0
    while i < len(all):
        alllist.append(all[i])
        t=1
        now_line=i
        while (all[now_line][1]==all[(now_line+t)%list_len][1]):
            t+=1
            i+=1
            alllist.append(all[i])
        #如果UIE提取的原因少于5个，随机生成缺少的负样本
        #负样本特征向量1范围（0，0.04），特征向量2范围（0，0.01）
        if f_num-t >0:
            for j in range(1,f_num-t+1):
                tmp=[]
                tmp.append(-1)
                tmp.append(all[now_line][1])
                tmp.append(random.random()*0.04)
                tmp.append(random.random()*0.01)
                alllist.append(tmp)
        i=i+1
    return alllist

#构建input格式  eg:  -1 qid:0 1:0.1 2:0.09
def form_input_data(alllist):
    allline=[]
    all_list=[]
    train_list=[]
    test_list=[]
    cott=-1
    len_data=len(alllist)
    for i in alllist:
        str1 = "" + str(i[0]) +" " +"qid:" + str(i[1])+" 1:"+str(i[2])+" 2:"+str(i[3])
        allline.append(str1)
        cott+=1
        all_list.append(str1)
        if cott<len_data*0.7:
            train_list.append(str1)
            # print(str1,file=mylog2)
        if cott>=len_data*0.7:
            test_list.append(str1)
            # print(str1,file=mylog3)
    return all_list,train_list,test_list   


#args.type:业绩归因
#将UIE提取的原因保存在test_reason.log中
def write_reason(args,data):
    cot=0
    for j in data[args.type]:
        log4.info(j["text"])
        cot+=1
    while cot<5:
        log4.info("Virtual_sample")
        cot+=1



#读取词库中的词
def read_word(word_fath):
    vocab=[]
    with open(word_fath, "r", encoding="utf8") as f1:
        words = f1.readlines()
        for i in words:
            vocab.append(i.strip('\n'))
    return vocab

#构建模型所需的数据,返回 全部数据列表，训练数据列表，测试数据列表，csv所需要的content和lab列表
def form_input_list(args):
    vocab = read_word("word.log")
    with open("my_ie1018v2.log", "r", encoding="utf8") as f:
        all_rows=[]  #
        pro_cnt=[]  #存词库中的词在原因中出现的次数
        key_num=-1  #标记quary
        txt_data=[] #生成的csv所需要的content,lab列表
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i][0] != '{':
                continue
            data = eval(lines[i])
            if len(data) == 0:  # 预测为无因果
                continue
            elem_num=len(data[args.type])        
            if elem_num>1:           # 预测出的因果不少于1个
                str2=lines[i-1][7:]
                data2=eval(str2)
                sentence_txt=data2["content"]
                lab_txt=data2["result_list"][0]["text"] #当前quary下管院标记的原因
                if len(lab_txt) != 0:
                    txt_data_line=[] 
                    txt_data_line.extend([sentence_txt, lab_txt])
                    txt_data.append(txt_data_line)
                    key_num+=1 #合法quary
                    line_elem=[]
                    write_reason(args,data)
                    for j in data[args.type]:
                        elem=[]#当前quary 下每一个原因的特征向量
                        tmp=string_similar(j["text"],lab_txt) #管院标记原因与UIE提取原因的相似度
                        #elem 每一维的含义
                        # 第0维：key_num  第一维：相似度  第二维：概率  第三维：词出现的次数
                        # 第四维：label（1，-1）  第5维：归一后的出现次数
                        cot = calculate_cot(j,vocab)
                        elem.extend([key_num, tmp, j["probability"], cot])                    
                        pro_cnt.append(cot)
                        line_elem.append(elem)
                    all_rows = generate_label(line_elem,all_rows) 
        cnt = normalize(pro_cnt)
        list_len=len(all_rows)
        all = form_input_vector(all_rows,cnt)
        alllist = generate_negative_sample_and_addin(all,list_len)
        all_list,train_list,test_list = form_input_data(alllist)
    return all_list, train_list, test_list, txt_data

def cmp(x,y):
    if x>y:
        return 1
    if x<y:
        return -1
    else:
        return 0

#解析有标签的数据文本，返回元组: (map[qid]feature_vec, list[qid])
def parse_labeled_data_file(fin):
    data = {}
    keys = []
    last_key = ""
    for line in fin:
        line = line.split("#")[0]
        # 1 qid:0 1:0.69 2:0.55
        elems = line.split(" ")
        label = float(elems[0])
        qid = elems[1].split(":")[1]
        feature_v = [0.0] * config.FEATURE_NUM
        # 提取line中的feature_v[1],feature_v[2]
        for i in range(2, config.FEATURE_NUM + 2):
            subelems = elems[i].split(":")
            # 1   0.69
            if len(subelems) < 2:
                continue
            index = int(subelems[0]) - 1
            feature_v[index] = float(subelems[1])
        # 
        if qid in data:
            data[qid].append([label] + feature_v)
        else:
            data[qid] = [[label] + feature_v]
        if last_key != qid:
            last_key = qid
            keys.append(qid)
    return data, keys

#计算需要一次检索的样本对
def calc_query_doc_pairwise_data(doc_list):
    """Calc required sample pairs from one retrival
    
    If doc_list contains A, B, C and A > B > C
         pairs are generated as A > B, A > C, B > C

    Args:
        doc_list: list of list: [score, f1, f2 , ..., fn]

    Returns:
        [X1, X2], [Y1, Y2]:
        X1.shape = X2.shape = (None, config.FEATURE_NUM)
        Y1.shape = Y2.shape = (None, 1)
    """
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    # sorted_doc_list = sorted(doc_list, cmp=lambda x, y: cmp(y[0], x[0]))
    sorted_doc_list = sorted(doc_list, key=functools.cmp_to_key(lambda x, y: cmp(y[0], x[0])))
    for i in range(len(sorted_doc_list)):
        for j in range(i + 1, len(sorted_doc_list), 1):
            X1.append(sorted_doc_list[i][1:])
            Y1.append(sorted_doc_list[i][0:1])
            X2.append(sorted_doc_list[j][1:])
            Y2.append(sorted_doc_list[j][0:1])
    return [X1, X2], [Y1, Y2]




if __name__ == "__main__":
    log1=get_logger('train')
    log2=get_logger('test')
    log3=get_logger('all_data')
    log4=get_logger('reason_of_test')

    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--type', type=str, default='业绩归因',help='原因类型')
    args = parser.parse_args()

    all_list, train_list, test_list, txt_data = form_input_list(args)
    all_list_len=len(all_list)
    for i in range(all_list_len):
        if i<int(all_list_len*0.7):
            log1.info(all_list[i])
        else:
            log2.info(all_list[i])
        log3.info(all_list[i])

    #将回答和label写入csv文件
    fd = open('1203_all_content.csv', 'w',encoding="utf-8-sig")
    writer = csv.writer(fd)
    header=['content','label']
    writer.writerow(header)
    csv_len=len(txt_data)
    for i in range(csv_len):
        if i>=csv_len*0.7:
            writer.writerow(txt_data[i])
