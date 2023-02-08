import logging
import argparse
from datetime import datetime
from bert import sentence_features, BertTextNet
from transformers import BertTokenizer
from rank_data_process import get_logger3, print_list, read_list

def add_embedding(args, uie_list):
    textNet = BertTextNet(args.code_length)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)        
    lines = uie_list 
    after_embedding_list = []
    for i in range(len(lines)):
        if lines[i][0] != '{':
            continue
        data_pre = eval(lines[i])
        dic = data_pre
        if len(data_pre["output"][0]) == 0:  # 预测为无因果
            log.info(dic)
            continue
        data=data_pre["output"][0]
        #data: {'业绩归因': [{'text': '“调整年”', 'start': 3, 'end': 8, 'probability': 0.6058174246136119}]}
        elem_num=len(data[args.type])                 
        if elem_num>0:  #至少预测出一个归因  
            dic_new = {}  
            uie_re = []
            str_pre = data_pre["content"]
            s_cls = '[CLS]'
            s_sep = '[SEP]'
            #对每个归因，构造上下文字符串
            for j in data[args.type]:           
                s_before = s_cls + str_pre[0 : j["start"]] + s_sep
                s_after = s_cls + str_pre[j["end"] : ] + s_sep
                sen_f = sentence_features(textNet, tokenizer, [s_before, s_after])
                j['s_before'] = sen_f[0]
                j['s_after'] = sen_f[1]
                uie_re.append(j)
            dic_new[args.type] = uie_re
        dic['output'] = [dic_new]
        after_embedding_list.append(dic)
    return after_embedding_list
               
                    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='embedding')
    parser.add_argument('--type', type=str, default='业绩归因',help='原因类型')
    parser.add_argument('--vocab_path', type=str, default='/data/fkj2023/Project/eccnlp/phrase_rerank/bert_model/vocab.txt',help='vocab path')
    parser.add_argument('--code_length', type=int, default=16,help='the dimension of sentence features')
    args = parser.parse_args()

    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/test.txt'
    filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/info_extraction_result_1222.txt'
    # filepath = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/info.txt'

    logpath = "/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/"  
    log = get_logger3('add_embedding', logpath)

    uie_list = read_list(filepath)
    after_embedding_list = add_embedding(args, uie_list)
    print_list(after_embedding_list, log)

