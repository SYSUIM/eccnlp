import functools
import difflib
import numpy as np
import logging
import math
import argparse
from datetime import datetime
from transformers import BertTokenizer
import torch
from transformers import BertModel, BertConfig,BertTokenizer
from torch import nn


config_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/bert_model/config.json'
model_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/bert_model/pytorch_model.bin'
vocab_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/bert_model/vocab.txt'


def read_list_file(path: str) -> list:
    data_list = []
    with open(path, 'r') as f:
        for line in f:
            data_list.append(eval(line.strip('\n'), {'nan': ''}))
    # logging.info(f'read {path} DONE. Length: {len(data_list)}')

    return data_list

def add_embedding(args, uie_list):
    textNet = BertTextNet(args.code_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    textNet.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)        
    lines = uie_list 
    after_embedding_list = []
    for i in range(len(lines)):
        if i % 1000 ==0:
            logging.info(f'add embedding for step {i}')
        data_pre = lines[i]
        dic = data_pre
        # no uie reason
        if len(data_pre["output"][0]) == 0: 
            after_embedding_list.append(dic)
            continue
        data=data_pre["output"][0]
        elem_num=len(data[args.type])   
        # at least one uie reason              
        if elem_num > 0: 
            dic_new = {}  
            uie_re = []
            str_pre = data_pre["content"]
            s_cls = '[CLS]'
            s_sep = '[SEP]'
            for j in data[args.type]:
                lb = 0
                re = -1    
                if j['start'] > 100 :
                    lb = j['start'] -100  
                if len(str_pre) - j["end"] >100:
                    re = j["end"] + 100
                s_before = s_cls + str_pre[lb : j["start"]] + s_sep
                s_after = s_cls + str_pre[j["end"] : re] + s_sep
                sen_f = sentence_features(textNet, tokenizer, [s_before, s_after], device)
                j['s_before'] = sen_f[0]
                j['s_after'] = sen_f[1]
                uie_re.append(j)
            dic_new[args.type] = uie_re
        dic['output'] = [dic_new]
        # log.info(dic)
        after_embedding_list.append(dic)
    return after_embedding_list

'''
TODO if the finetuned BERT classification parameters could be reused here?
'''
class BertTextNet(nn.Module):
    def __init__(self,code_length):
        super(BertTextNet, self).__init__()
 
        modelConfig = BertConfig.from_pretrained(config_path)
        self.textExtractor = BertModel.from_pretrained(
            model_path, config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size
 
        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()
 
    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)
 
        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features

'''
TODO 
1. tokenization faster
2. parallel compute using multiple GPUs
'''
def sentence_features(textNet, tokenizer, texts, device):
    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)  
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    max_len = max([len(single) for single in tokens]) 

    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding

    tokens_tensor = torch.tensor(tokens)
    segments_tensors = torch.tensor(segments)
    input_masks_tensors = torch.tensor(input_masks)

    tokens_tensor, segments_tensors, input_masks_tensors = tokens_tensor.to(device), segments_tensors.to(device), input_masks_tensors.to(device)

    text_hashCodes = textNet(tokens_tensor, segments_tensors, input_masks_tensors)  # text_hashCodes is a 16-dim text feature

    return text_hashCodes.tolist()


def get_text_list(uie_list):
    text_list = []
    num_list = []
    lines = uie_list      
    for i in range(len(lines)):
        data_pre = lines[i]
        if (data_pre["number"] not in num_list):
            num_list.append(data_pre["number"])
            text_list.append(data_pre["raw_text"])
    return text_list, num_list

# O(N)
def merge_reasons(args, text_list, num_list, uie_list):
    
    merged_reasons = []

    # qid_map: key: value of number
    #          value： number index in uie_list
    qid_map = {}
    idx = 0
    for record in uie_list:
        qid_map.setdefault(record['number'], [])
        qid_map[record['number']].append(idx)
        idx += 1

    for num in qid_map.keys():  #  iterate over number 
        id_list = qid_map[num]  # a list of many short setences' indexes in uie_list
        dic = {}
        reasons = []
        reason_set = set()
        res_list = []
        dic_rea={}
        for i in id_list:   # iterate over short setences' indexes of a qid
            data_pre = uie_list[i]
            prompt = data_pre["prompt"]
            dic_rea[args.type] = reasons
            if data_pre["result_list"][0]["text"] != '':
                res_list.append(data_pre["result_list"][0])            
            data = data_pre["output"][0]
            if len(data) == 0 :
                continue 
            for k in data[args.type]:  # k: every uie reason (dict)
                if k['text'] not in reason_set:
                    reasons.append(k)
                    reason_set.add(k['text'])
        dic['raw_text'] = text_list[num_list.index(num)]
        dic['number'] = num
        dic['result_list'] = res_list
        dic['prompt'] = prompt
        dic['output'] = [dic_rea]
        # log.info(dic)
        merged_reasons.append(dic)

    return merged_reasons


def print_list(alist, log):
    for i in alist:
        log.info(i)

def get_logger1(name,logpath):
    logger = logging.getLogger(name)
    filename = f'{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}_{name}.log'
    fh = logging.FileHandler(logpath + filename, mode='w+', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def get_logger2(name, logpath):
    logger = logging.getLogger(name)
    filename = f'{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}_{name}.log'
    fh = logging.FileHandler(logpath + filename, mode='a+', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    logger.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

'''
TODO move to utils.py
'''
def read_word(filepath):
    alist = []
    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            alist.append(line)
    return alist

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

# define sort
# 1：similarity
# 2：uie probability
# 3: word frequency
def list_cmp(x,y):
    if x[1] != y[1]:
        return x[1]>y[1]
    elif x[3] != y[3]:
        return x[3]>y[3]
    else:
        return x[2]>y[2]


def normalize(a_list):
    y=[]
    arr = np.asarray(a_list)
    for x in arr:
        x = float(x - np.min(arr))/(np.max(arr)- np.min(arr))
        y.append(x)
    return y


def calculate_cot(uie_reason,vocab):
    cot=0
    for search_list in vocab:
        if search_list in uie_reason:
            cot=cot+1
    return cot
   
# rank1 :3  rank2:  2  rank3: 1  rank4: 0  after: 0          
def generate_label(line_elem,all_rows):
    line_elem.sort(key=functools.cmp_to_key(list_cmp))
    for j in range(len(line_elem)):
        if j==0:
            line_elem[j].append(3)
        elif j == 1:
            line_elem[j].append(2)
        elif j == 2:
            line_elem[j].append(1)
        else:
            line_elem[j].append(0)
        all_rows.append(line_elem[j])        
    return all_rows


def form_input_vector(all_rows,cnt):
    y=[] 
    list_len=len(all_rows)
    line_len = len(all_rows[0])
    for i in range(list_len):
        line_in_all=[]
        line_in_all.append(all_rows[i][line_len - 1]) #label
        line_in_all.append(all_rows[i][0]) # key_num
        line_in_all.append(all_rows[i][2]) # uie probability
        line_in_all.append(cnt[i])         # word frequency
        line_in_all = line_in_all + all_rows[i][4 : line_len - 1] # context features
        y.append(line_in_all)
    return y


# train_list: test_list = 7:3
def form_input_data(args, alllist, reason_list):
    all_list=[]
    train_list=[]
    test_list=[]
    reason_of_test=[]
    len_data=len(alllist)
    qid_num = alllist[len_data - 1][1] + 1
    bre = int(math.floor(qid_num*0.7))
    for i in alllist:
        all_list.append(i)
        if i[1] < bre :
            train_list.append(i)
        if i[1] >= bre:
            test_list.append(i)
            reason_of_test.append(reason_list[alllist.index(i)])

    return all_list,train_list,test_list,reason_of_test 

# write uie reasons into reason_list
def write_reason(args, data, reason_list):
    for j in data[args.type]:
        reason_list.append(j["text"])

def read_word(word_fath):
    vocab=[]
    with open(word_fath, "r", encoding="utf8") as f1:
        words = f1.readlines()
        for i in words:
            vocab.append(i.strip('\n'))
    return vocab

def form_input_list(args, merged_list, vocab):
    all_rows=[]  
    pro_cnt=[]  # word cnt
    reason_list = []
    key_num=-1  
    lines = merged_list
    for i in range(len(lines)):
        data_pre = lines[i]
        if len(data_pre["output"][0]) == 0:  
            continue
        data=data_pre["output"][0]
        elem_num=len(data[args.type])        
        if elem_num>1:           # number of uie reason >1
            lab_txt=data_pre["result_list"][0]["text"]   #  label text 
            if len(lab_txt) != 0:
                key_num+=1 #合法quary
                line_elem=[]
                write_reason(args, data, reason_list)                   
                for j in data[args.type]:
                    elem=[]  # all features
                    tmp=string_similar(j["text"],lab_txt) 
                    #elem 每一维的含义
                    # 第0维：key_num  第一维：相似度  第二维：概率  第三维：词出现的次数
                    # 第四维：label（1，-1）  第5维：归一后的出现次数
                    cot = calculate_cot(j["text"],vocab)
                    elem.extend([key_num, tmp, j["probability"], cot])
                    elem = elem + j["s_before"] + j["s_after"]              
                    pro_cnt.append(cot)
                    line_elem.append(elem)
                all_rows = generate_label(line_elem,all_rows) 
    cnt = normalize(pro_cnt)
    list_len=len(all_rows)
    all = form_input_vector(all_rows,cnt)
    all_list,train_list,test_list,reason_of_test = form_input_data(args, all, reason_list)
    return all_list, train_list, test_list, reason_of_test


def form_predict_input_list(args, merged_list, vocab):
    all_rows=[]  
    pro_cnt=[]  
    key_num=-1 
    lines = merged_list
    reasons=[] 
    for i in range(len(lines)):
        data_pre = lines[i]
        if len(data_pre["output"][0]) == 0:  
            continue
        data=data_pre["output"][0]
        elem_num=len(data[args.type])                   
        if elem_num > 0:          
            key_num+=1 
            line_elem=[]             
            for j in data[args.type]:
                reasons.append(j["text"])
                elem=[]
                tmp = 0
                cot = calculate_cot(j["text"],vocab)
                elem.extend([key_num, tmp, j["probability"], cot])  
                elem = elem + j["s_before"] + j["s_after"]                  
                pro_cnt.append(cot)
                line_elem.append(elem)
            all_rows = generate_label(line_elem,all_rows) 
    cnt = normalize(pro_cnt)
    all = form_input_vector(all_rows,cnt)
    all_list,train_list,test_list,useless_reason= form_input_data(args, all, reasons)
    return all_list, reasons


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--type', type=str, default='业绩归因',help='type of answer')
    parser.add_argument('--reason_num', type=int, default=10,help='reason number')
    # parser.add_argument('--path_of_merged_reasons', type=str, default='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2.0_2023-01-15_merge.txt',help='path of merged reasons')
    parser.add_argument('--f_num', type=int, default=34, help='feature number')
    parser.add_argument('--usage', type=str, default="train", help='choose train or predict')
    
    args = parser.parse_args()

    vocab = read_word("/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/2022-12-24_word.log")

    logpath = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/transfer/" 

    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2.0_2023-01-15_merge.txt'
    filepath ='/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/merged/2023-03-02_merged_list.log'
    merged_list = read_list(filepath)

    if (args.usage == "train"):
        log1=get_logger1('train', logpath)
        log2=get_logger1('test', logpath)
        log3=get_logger1('all_data', logpath)
        log4=get_logger1('reason_of_test', logpath)

        all_list, train_list, test_list, reason_of_test = form_input_list(args, merged_list, vocab)

        for i in range(len(train_list)):
            log1.info(train_list[i]) 
        for i in range(len(test_list)):
            log2.info(test_list[i])
        for i in range(len(all_list)):
            log3.info(all_list[i])
        for i in range(len(reason_of_test)):
            log4.info(reason_of_test[i])

    if (args.usage == "predict"):
        log5=get_logger1('all_predict_data', logpath)
        log6=get_logger1('all_predict_reasons', logpath)
        
        all_list, reasons = form_predict_input_list(args, merged_list, vocab)

        for i in range(len(all_list)):
            log5.info(all_list[i]) 

        for i in range(len(reasons)):
            log6.info(reasons[i]) 
